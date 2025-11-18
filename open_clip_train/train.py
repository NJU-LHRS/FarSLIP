import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast

from torchvision.ops import roi_align
from torchvision.transforms import RandomResizedCrop

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.total_sum =0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.total_sum += val * n


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def filter_batch_data(batch, used_losses, input_dtype, device):
    selected = {}

    selected["global_image"] = batch["global_image"].to(device=device, dtype=input_dtype, non_blocking=True)
    if 'global_itc' in used_losses:
        selected["global_text"] = batch["global_text"].to(device=device, non_blocking=True)
    if 'local_itc' in used_losses:
        selected["local_images"] = batch["local_images"].to(device=device, dtype=input_dtype, non_blocking=True)
        selected["local_texts"] = batch["local_texts"].to(device=device, non_blocking=True)
        selected["boxes"] = batch["boxes"].to(device=device, non_blocking=True)
        if 'local_categories' in batch:
            selected["local_categories"] = batch["local_categories"].to(device=device)
    if 'distill' in used_losses and 'subset_images' in batch:
        selected["subset_images"] = batch["subset_images"].to(device=device, dtype=input_dtype, non_blocking=True)
        selected["subset_boxes"] = batch["subset_boxes"].to(device=device, non_blocking=True)

    return selected


def get_batch_data(batch, used_losses, input_dtype, device):
    images, global_caption = batch

    global_image = torch.stack([item["global"] for item in images], dim=0)  # (B, C, H, W)
    local_imgs = torch.stack([item["locals"] for item in images], dim=0)  # (B, N, C, H_l, W_l)
    bboxes = torch.stack([item["bboxes"] for item in images], dim=0)  # (B, N, 5)

    selected = {}
    selected["global_image"] = global_image.to(device=device, dtype=input_dtype, non_blocking=True)
    selected["global_text"] = global_caption.to(device=device, non_blocking=True)
    if 'distill' in used_losses:
        selected["subset_images"] = local_imgs.to(device=device, dtype=input_dtype, non_blocking=True)
        selected["subset_boxes"] = bboxes.to(device=device, non_blocking=True)

    return selected


def train_one_epoch(model, teacher, method, data, loss, mpcl_loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if teacher: teacher.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq      # Accumulated gradient updates at batch (i)
        step = num_batches_per_epoch * epoch + i_accum      # Total steps for whole training

        if not args.skip_scheduler:
            scheduler(step)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if method.startswith('farslip'):
            assert args.accum_freq == 1, "accum freq disabled"

            # ========== Obtain input batches ==========
            if method == 'farslip1':
                batch = get_batch_data(batch, input_dtype=input_dtype, used_losses=args.loss_type, device=device)
            elif method == 'farslip2':
                batch = filter_batch_data(batch, input_dtype=input_dtype, used_losses=args.loss_type, device=device)
            else: raise ValueError(f"Unknown args.method: {args.method}")

            # ========== Obtain VALID local inputs (images, boxes, categories) ==========
            if "local_itc" in args.loss_type:

                bboxes = batch["boxes"]  # (B, num_boxes, 5)
                local_images = batch["local_images"]  # (B, num_boxes, C, H, W)
                local_texts = batch["local_texts"]  # (B, num_boxes, context_length)

                local_texts_list = []
                bboxes_list, local_imgs_list = [], []
                local_categories_list = []  # (B, num_boxes)

                has_local_categories = 'local_categories' in batch
                if has_local_categories: local_categories = batch['local_categories']

                for idx in range(len(bboxes)):
                    bbox = bboxes[idx]
                    local_img = local_images[idx]
                    validity = bbox[:, -1] == 1
                    bboxes_list.append(bbox[validity, :4])
                    local_imgs_list.append(local_img[validity])
                    local_text = local_texts[idx]
                    local_texts_list.append(local_text[validity])

                    if has_local_categories: local_categories_list.append(local_categories[idx][validity])

                local_texts = torch.cat(local_texts_list)  # (valid_objects, context_length)
                batch["local_texts"] = local_texts

                local_images = torch.cat(local_imgs_list)  # (valid_objects, C, H, W)
                batch['local_images'] = local_images

                if has_local_categories: local_categories = torch.cat(local_categories_list)

            if "distill" in args.loss_type and "subset_images" in batch:

                batch['subset_images'] = batch['subset_images'].reshape(-1, *batch['subset_images'].shape[2:])
                subset_bboxes_list = [bbox[:, :4] for bbox in batch["subset_boxes"]]

            # ============= Start training ==============

            def pad_tensor(tensor, total_len):
                valid_len, feat_dim = tensor.shape
                padded = torch.zeros((total_len, feat_dim), device=tensor.device, dtype=tensor.dtype)
                padded[:valid_len] = tensor
                return padded, valid_len, feat_dim

            visual_backbone = model.module.visual if hasattr(model, 'module') else model.visual
            grid_size = visual_backbone.grid_size   

            with autocast():
                features = model(batch=batch, device=device, input_dtype=input_dtype, used_losses=args.loss_type, last_attn_type=args.last_attn_type)
                if teacher and "distill" in args.loss_type:
                    with torch.no_grad():
                        features_t = teacher(batch=batch, device=device, input_dtype=input_dtype, used_losses=args.loss_type, last_attn_type=args.last_attn_type)
                logit_scale = features["logit_scale"]

                def _denormalize_boxes(normed_boxes, x):
                    h, w = x.shape[-2:]
                    denormed_boxes = []
                    for boxes in normed_boxes:
                        new_boxes = boxes.clone()
                        new_boxes[:, [0, 2]] *= w
                        new_boxes[:, [1, 3]] *= h
                        denormed_boxes.append(new_boxes)
                    return denormed_boxes

                def extract_normed_roi_features(global_patches, bboxes_list, grid_size):
                    """
                    Extract ROI features from global_patches, then perform mean pooling and normalize.
                    """
                    B, N, D = global_patches.shape  # (B, N_patches, D)
                    H, W = grid_size

                    patches_2d = global_patches.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
                    rois = _denormalize_boxes(bboxes_list, patches_2d)  # (B*n, 5)
                    roi_feats = roi_align(patches_2d, rois, output_size=grid_size, spatial_scale=1.0, sampling_ratio=-1,
                                          aligned=True)
                    # roi_global_features = F.normalize(roi_feats, dim=-1)
                    pooled = roi_feats.mean(dim=[2, 3])                         # (B*n, D)
                    normed = F.normalize(pooled, dim=-1)
                    return normed

                # ============== feat from global patches ===============
                if "distill" in args.loss_type:
                    normed_pooled_roi_global_features = extract_normed_roi_features(
                        features["global_patches"], subset_bboxes_list, grid_size)

                losses = {}
                # ============ distill loss ============
                if "distill" in args.loss_type:     # ["roi2cls", "roi2pooled", "combined"]
                    if args.distill_align == 'roi2pooled':      # patch-to-patch alignment
                        if args.distill_type == 'active':
                            local_patches_t = features["subset_patches"]
                        else: local_patches_t = features_t["subset_patches"]

                        local_patches_t = local_patches_t.view(local_patches_t.shape[0], grid_size[0], grid_size[1],
                                                           -1).permute(0, 3, 1, 2)
                        pooled_local_patches_t = local_patches_t.mean(dim=[2, 3])
                        normed_pooled_local_patches_t = F.normalize(pooled_local_patches_t, dim=-1)
                        loss_cosine = 1.0 - (normed_pooled_roi_global_features * normed_pooled_local_patches_t).sum(-1).mean()    # global roi & local roi

                    elif args.distill_align == 'roi2cls':       # indirect alignment with text
                        if args.distill_type=='active':
                            normed_local_pooled_t = F.normalize(features["subset_image_pooled"], dim=-1)
                        else:
                            normed_local_pooled_t = F.normalize(features_t["subset_image_pooled"], dim=-1)
                        loss_cosine = 1.0 - (normed_pooled_roi_global_features * normed_local_pooled_t).sum(-1).mean()  # global roi & local pooled

                    elif args.distill_align == 'combined':
                        if args.distill_type == 'active':
                            local_patches_t = features["subset_patches"]
                            local_image_pooled_t = features["subset_image_pooled"]
                        else:
                            local_patches_t = features_t["subset_patches"]
                            local_image_pooled_t = features_t["subset_image_pooled"]

                        local_patches_t = local_patches_t.view(local_patches_t.shape[0], grid_size[0], grid_size[1],
                                                           -1).permute(0, 3, 1, 2)
                        pooled_local_patches_t = local_patches_t.mean(dim=[2, 3])
                        normed_pooled_local_patches_t = F.normalize(pooled_local_patches_t, dim=-1)

                        normed_local_pooled_t = F.normalize(local_image_pooled_t, dim=-1)

                        loss_roi2roi = 1.0 - (normed_pooled_roi_global_features * normed_pooled_local_patches_t).sum(-1).mean()
                        loss_roi2pooled = 1.0 - (normed_pooled_roi_global_features * normed_local_pooled_t).sum(-1).mean()
                        loss_cosine = 0.5 * loss_roi2roi + 0.5 * loss_roi2pooled
                    else:
                        raise ValueError(f"Unknown distill_align: {args.distill_align}")
                    losses["distill"] = loss_cosine

                # ============ local_itc loss ============
                if "local_itc" in args.loss_type:       # ["cls", "pooled", "roi"]
                    total_len = args.batch_size * args.max_boxes

                    if args.local_itc_align == "cls":
                        normed_local_pooled = F.normalize(features["local_image_pooled"], dim=-1)
                        local_feat = normed_local_pooled

                    elif args.local_itc_align == "pooled":
                        local_patches = features["local_patches"]
                        local_patches = local_patches.view(local_patches.shape[0], grid_size[0], grid_size[1],
                                                           -1).permute(0, 3, 1, 2)
                        pooled_local_patches = local_patches.mean(dim=[2, 3])
                        normed_pooled_local_patches = F.normalize(pooled_local_patches, dim=-1)

                        local_feat = normed_pooled_local_patches

                    elif args.local_itc_align == "roi":
                        local_feat = extract_normed_roi_features(
                            features["global_patches"], bboxes_list, grid_size)

                    else:
                        raise ValueError(f"Unknown local_itc_align: {args.local_itc_align}")

                    padded_normed_local_pooled, valid_len, feature_len = pad_tensor(local_feat, total_len)

                    normed_local_text_pooled = F.normalize(features["local_text_pooled"], dim=-1)
                    padded_normed_local_text_pooled, _, _ = pad_tensor(normed_local_text_pooled, total_len)

                    padding_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
                    padding_mask[:valid_len] = True

                    padded_local_categories = torch.full((total_len,), -1, dtype=local_categories.dtype,
                                                         device=local_categories.device)
                    padded_local_categories[:valid_len] = local_categories

                    if mpcl_loss:
                        region_itc_loss = mpcl_loss(padded_normed_local_pooled, padded_normed_local_text_pooled, logit_scale, padded_local_categories)
                    else:
                        region_itc_loss = loss(padded_normed_local_pooled, padded_normed_local_text_pooled, logit_scale, padding_mask=padding_mask)

                    losses["region_itc"] = region_itc_loss

                # ============ global_itc loss ============
                if "global_itc" in args.loss_type:
                    normed_global_pooled = F.normalize(features["global_image_pooled"], dim=-1)
                    normed_global_text_pooled = F.normalize(features["global_text_pooled"], dim=-1)
                    assert normed_global_pooled.shape[0] == normed_global_text_pooled.shape[0]
                    global_itc_loss = loss(normed_global_pooled, normed_global_text_pooled, logit_scale)
                    losses["global_itc"] = global_itc_loss

                total_loss = losses.get("distill", 0) * args.w_d + losses.get("region_itc", 0) * args.w_l + losses.get("global_itc", 0) * args.w_g
                losses["loss"] = total_loss

            backward(total_loss, scaler)


        elif method == 'vanilla':
            # images, texts = batch
            images, texts = batch["global_image"], batch["global_text"]
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            if args.accum_freq == 1:
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out["logit_scale"]
                    losses = loss(**model_out, output_dict=True)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)
            else:
                # First, cache the features without any gradient tracking.
                with torch.no_grad():
                    with autocast():
                        model_out = model(images, texts)

                        for f in ("logit_scale", "logit_bias"):
                            model_out.pop(f, None)

                        for key, val in model_out.items():
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                    accum_images.append(images)
                    accum_texts.append(texts)

                # If (i + 1) % accum_freq is not zero, move on to the next batch.
                if ((i + 1) % args.accum_freq) > 0:
                    # FIXME this makes data time logging unreliable when accumulating
                    continue

                # Now, ready to take gradients for the last accum_freq batches.
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                # Call backwards each time, but only step optimizer at the end.
                optimizer.zero_grad()
                for j in range(args.accum_freq):
                    images = accum_images[j]
                    texts = accum_texts[j]
                    with autocast():
                        model_out = model(images, texts)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                        losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        del inputs
                        del inputs_no_accum
                        total_loss = sum(losses.values())
                        losses["loss"] = total_loss

                    backward(total_loss, scaler)

        if teacher and args.distill_type == 'ema':
            # EMA update for the teacher
            momentum = args.EMA_momentum
            with torch.no_grad():
                for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                    param_k.data.mul_(momentum).add_(
                        (1 - momentum) * param_q.detach().data)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            if teacher and args.distill_type == 'ema': unwrap_model(teacher).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = dataloader.batch_size
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            eta_total = batch_time_m.total_sum / (i+1) * (args.epochs - epoch - (i+1)/dataloader.num_batches) * dataloader.num_batches
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                # f"ETA: {eta_total:.3f}s "
                f"ETA: {int(eta_total // 3600):02}h{int((eta_total % 3600) // 60):02}m{int(eta_total % 60):02}s "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
