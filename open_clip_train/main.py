import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # adjust args
    if args.force_quick_gelu: args.wandb_tags = ['qg'] + args.wandb_tags
    loss_str = ''.join(word[0].upper() for word in args.loss_type)
    args.wandb_tags = [f"l_{loss_str}"] + args.wandb_tags
    if args.long_clip == 'disable': args.wandb_tags = ['VC'] + args.wandb_tags      # vanilla CLIP
    elif args.long_clip in ["load_from_clip", "load_from_scratch"]: args.wandb_tags = ['LC'] + args.wandb_tags  # Long-CLIP
    else: raise ValueError('Wrong long_clip in args')
    # if args.mpcl_loss and 'local_itc' in args.loss_type: args.wandb_tags = args.wandb_tags + ['mpcl']
    if args.frozen_text: args.wandb_tags = args.wandb_tags + ['ft']
    if args.method == 'farslip1':
        if 'local_itc' in args.loss_type: raise ValueError(f'Local_itc cannot be activated for farslip1.')
        # args.use_imagecrop_aug = True

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join(args.wandb_tags+[
            date_str,
            f"d_{args.train_dataset_name}",
            f"{model_name_safe}",
            f"lr_{args.lr}",
            f"wd_{args.wd}",
            f"b_{args.batch_size}",
            f"e_{args.epochs}",
            f"w_{args.world_size}",
            # f"j_{args.workers}",
            # f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    args.logs = os.path.join(args.logs, args.model)
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    args.distill = False    # We remove the original distill implemented in OpenCLIP

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)

    # Load model
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        long_clip=args.long_clip,

        use_imagecrop_aug = args.use_imagecrop_aug,
        max_boxes = args.max_boxes,
        local_method= args.local_method,
        **model_kwargs,
    )
    if args.long_clip == 'load_from_clip':
        model.load_from_pretrained_short_pe(keep_len=20)

    if args.frozen_text:
        for module in [model.transformer, model.ln_final]:
            for param in module.parameters():
                param.requires_grad = False

        if isinstance(model.text_projection, torch.nn.Module):
            for param in model.text_projection.parameters():
                param.requires_grad = False
        elif isinstance(model.text_projection, torch.Tensor):
            model.text_projection.requires_grad = False

    if 'distill' in args.loss_type and args.distill_type != "active":
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad = False
    else: teacher = None

    if args.distill_type == 'frozen':
        assert 'local_itc' not in args.loss_type and 'global_itc' not in args.loss_type, \
            "'frozen' distill_type cannot be used with local_itc or global_itc in loss_type"

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        if args.find_unused_parameters:
            ddp_args['find_unused_parameters'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.train_dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            if is_master(args):
                defaults = copy.deepcopy(optimizer.defaults)
                defaults['weight_decay'] = args.wd
                defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
                logging.info(
                    f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
                )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            sd_teacher = checkpoint.get("state_dict_teacher", None)

            if not args.distributed:
                if next(iter(sd.items()))[0].startswith('module.'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                if sd_teacher is not None and next(iter(sd_teacher.items()))[0].startswith('module.'):
                    sd_teacher = {k[len('module.'):]: v for k, v in sd_teacher.items()}

            model.load_state_dict(sd)
            if teacher is not None and sd_teacher is not None:
                print("Loading teacher state dict for resuming.")
                teacher.load_state_dict(sd_teacher)

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize tokenizer & datasets
    context_length = 77 if args.long_clip == 'disable' else 248        # Long-CLIP is supported through enabling args.long_clip
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=context_length)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        if args.train_data is not None:
            args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=f'{args.wandb_project_name}-{args.model}',
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
            dir=log_base_path
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    original_teacher = teacher
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)
        teacher = torch.compile(original_teacher)

    if 'train' not in data:
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    mpcl_loss = None
    if args.mpcl_loss:
        from open_clip.loss import MultiPosConLossMM
        mpcl_loss = MultiPosConLossMM(
            rank=args.rank,
            world_size=args.world_size,
            temperature=0.07, w1=1.0, w2=1.0
        )

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, teacher, args.method, data, loss, mpcl_loss, epoch, optimizer, scaler, scheduler, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if original_teacher is not None:
                checkpoint_dict["state_dict_teacher"] = original_teacher.state_dict()

            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    # main(sys.argv[1:])

    from open_clip_train.config import arg_dict

    cli_args = sys.argv[1:]

    arg_list = []
    for k, v in arg_dict.items():
        if v is None:
            arg_list.append(k)
        else:
            if isinstance(v, list):
                arg_list.append(k)
                arg_list.extend(map(str, v))
            else: arg_list.append(f"{k}={v}")

    combined_args = arg_list + cli_args
    main(combined_args)

    # main(arg_list)