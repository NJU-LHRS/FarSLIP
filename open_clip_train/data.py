import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import io

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from torchvision.transforms import RandomResizedCrop
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None



def get_json_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, root_img_dir=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        root_img_dir=root_img_dir,
        args=args
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def factor_pair(n):
    for i in range(int(math.isqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i
        

class JsonDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, root_img_dir=None, args=None):
        logging.debug(f'Loading json data from {input_filename}.')
        self.args = args
        self.max_boxes = args.max_boxes
        self.min_size = args.min_size
        self.max_size = args.max_size
        self.root_img_dir = root_img_dir

        with open(input_filename, 'r') as f:
            data = json.load(f)
        self.data_list = data

        self.transforms = transforms
        self.tokenize = tokenizer

        self.grid = factor_pair(args.max_boxes)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        ori_global_image = Image.open(os.path.join(self.root_img_dir, item['global_filepath']))
        global_image = self.transforms(ori_global_image)
        ori_global_caption = item.get("global_caption") or item.get("detailed_caption")
        # ori_global_caption = item.get("global_caption") or item.get("brief_caption")
        global_caption = self.tokenize(ori_global_caption)[0]

        # ====== Global/local image/text ======
        def expand_bbox(bbox, scale):
            x1, y1, x2, y2, W, H = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = (x2 - x1) * scale, (y2 - y1) * scale
            x1n = max(cx - w / 2, 0)
            y1n = max(cy - h / 2, 0)
            x2n = min(cx + w / 2, W)
            y2n = min(cy + h / 2, H)
            return x1n, y1n, x2n, y2n

        segments = item['segment']
        boxes_list, imgs_list, texts_list, cats_list = [], [], [], []

        indices = list(range(len(segments)))
        random.shuffle(indices)

        for box_id in indices:
            segment = segments[box_id]
            bbox = segment['bbox']
            x1, y1, x2, y2 = expand_bbox(bbox.values(), 1)
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_size ** 2 or (self.max_size and area > self.max_size ** 2):
                continue

            boxes_list.append([x1, y1, x2, y2, 1.0])
            crop = ori_global_image.crop((x1, y1, x2, y2))
            imgs_list.append(self.transforms(crop))

            if 'cap' in segment:
                texts_list.append(self.tokenize(segment['cap']).squeeze(0))
                cats_list.append(-1)
            else:
                texts_list.append(self.tokenize(f"a satellite image of {segment['category']}").squeeze(0))
                cats_list.append(segment['category_id'])

            if len(boxes_list) >= self.max_boxes:
                break

        # Padding to the length of max_boxes
        num_valid = len(boxes_list)
        boxes = torch.zeros(self.max_boxes, 5)
        local_imgs = torch.zeros(self.max_boxes, 3, *global_image.shape[-2:])
        local_texts = torch.zeros(self.max_boxes, global_caption.shape[-1], dtype=global_caption.dtype)
        local_categories = torch.full((self.max_boxes,), -1, dtype=torch.int32)

        if num_valid > 0:
            boxes[:num_valid] = torch.tensor(boxes_list)
            local_imgs[:num_valid] = torch.stack(imgs_list)
            local_texts[:num_valid] = torch.stack(texts_list)
            local_categories[:num_valid] = torch.tensor(cats_list)
        else:
            pass            # TODO: If no boxes, use augmentation

        _, h, w = global_image.shape
        scale = (h / ori_global_image.height, w / ori_global_image.width)
        boxes[:, [0, 2]] *= scale[1]
        boxes[:, [1, 3]] *= scale[0]
        boxes[:, [0, 2]] /= w           # => [0,1]
        boxes[:, [1, 3]] /= h

        out_dict = {
            "global_image": global_image,
            "global_text": global_caption,
            "boxes": boxes,
            "local_images": local_imgs,
            "local_texts": local_texts,
            "local_categories": local_categories
        }

        if self.args.local_method == 'objects':
            return out_dict

        # # ====== Random Crop ======
        elif self.args.local_method == 'randomcrops':

            boxes = torch.zeros(self.max_boxes, 5)  # x1, y1, x2, y2, validity
            local_imgs = torch.zeros(self.max_boxes, 3, *global_image.shape[-2:])
            width, height = ori_global_image.size
            for idx in range(self.max_boxes):
                i, j, h, w = RandomResizedCrop.get_params(ori_global_image, scale=(0.3, 0.7), ratio=(3 / 4, 4 / 3))
                x1 = j / width
                y1 = i / height
                x2 = (j + w) / width
                y2 = (i + h) / height
                boxes[idx] = torch.tensor([x1, y1, x2, y2, 1.0])
                local_imgs[idx] = self.transforms(ori_global_image.crop((j, i, j + w, i + h)))

        # ====== Regular Grid (CLIPSelf) ======
        elif self.args.local_method == 'grids':
            M, N = self.grid
            box_num = M * N
            grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, 1, N + 1),
                torch.linspace(0, 1, M + 1),
                indexing='xy'
            )
            x0y0s = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)  # [M,N,2]
            x1y1s = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)  # [M,N,2]
            boxes = torch.cat([torch.cat([x0y0s, x1y1s], dim=-1).view(-1, 4), torch.ones(M * N, 1)],
                              dim=1)  # [x1, y1, x2, y2, validity]

            local_imgs = torch.zeros(box_num, 3, *global_image.shape[-2:])
            width, height = ori_global_image.size
            for idx in range(box_num):
                x1, y1, x2, y2 = [int(boxes[idx][i] * (width if i % 2 == 0 else height)) for i in range(4)]
                local_imgs[idx] = self.transforms(ori_global_image.crop((x1, y1, x2, y2)))

        out_dict["subset_boxes"] = boxes
        out_dict["subset_images"] = local_imgs

        return out_dict


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None, root_img_dir=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        # change the image path
        if root_img_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: os.path.join(root_img_dir, x))

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    '''This is adapted based on OpenCLIP'''
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            raise RuntimeError(
                'Currently, the number of dataset samples must be specified for the training dataset. '
                'Please specify it via `--train-num-samples`.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    def byte_decode(x):
        return x.decode("utf-8")

    def custom_decoder(sample):
        if "img_content" in sample:
            img_bytes = sample["img_content"]
            sample["image"] = Image.open(io.BytesIO(img_bytes))
        if "img_name" in sample:
            sample["img_name"] = byte_decode(sample["img_name"])
        if "caption" in sample:
            sample["caption"] = byte_decode(sample["caption"])
        return sample

    pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        pipeline.extend([
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=args.seed,
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
        ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.map(custom_decoder),
        wds.map(lambda sample: {
            "image": sample["image"],
            "text": sample["caption"],
            # "name": sample["img_name"]
        }),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        num_shards = num_shards or len(expand_urls(input_shards)[0])
        assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    dataloader.batch_size = args.batch_size

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, root_img_dir=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        root_img_dir=root_img_dir,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    elif dataset_type == "json":
        return get_json_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.train_dataset_type == "synthetic":
        if args.train_dataset_type == "webdataset":
            data["train"] = get_dataset_fn(args.train_data, args.train_dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
        else:
            data["train"] = get_dataset_fn(args.train_data, args.train_dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer, root_img_dir=args.root_train_img_dir)


    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.val_dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer, root_img_dir=args.root_val_img_dir)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
