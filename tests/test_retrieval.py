from PIL import Image
import os
from os.path import join
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.precision import get_autocast


# Please specify the paths to data frames
# path to your csv files of text annotations.
DATA_CSV_PATH_DICT = {
    'SkyScript': '/path/to/your/skyscript/SkyScript_test_30K_filtered_by_CLIP_openai.csv',
    'RSICD': '/path/to/your/retrieval/RSICD/RSICD_img_txt_pairs_test.csv',
    'RSITMD': '/path/to/your/retrieval/RSITMD/RSITMD_img_txt_pairs_test.csv',
    'ucmcaptions': '/path/to/your/retrieval/ucmcaptions/ucmcaptions_img_txt_pairs_test.csv',
}
# path to your root of images (RSICD, RSITMD, and ucmcaptions share the same root dir.)
SKYSCRIPT_IMAGE_DIR = '/path/to/your/skyscript/'
RETRIEVAL_IMAGE_DIR = '/path/to/your/retrieval/'


batch_size = 128
precision = 'amp'
autocast = get_autocast(precision)


class CsvDataset_customized(Dataset):
    def __init__(self, df, transforms, img_key, caption_key, tokenizer=None, return_img_path=False,
                 root_data_dir=None):
        if root_data_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.tokenize = tokenizer
        self.return_img_path = return_img_path

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        if self.return_img_path:
            return images, texts, str(self.images[idx])
        return images, texts


class CsvDataset_image(Dataset):
    def __init__(self, df, transforms, img_key, return_img_path=False, root_data_dir=None):
        if root_data_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.images = df[img_key].tolist()
        self.transforms = transforms
        self.return_img_path = return_img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.return_img_path:
            return images, str(self.images[idx])
        return images


class CsvDataset_text(Dataset):
    def __init__(self, df, caption_key, tokenizer=None, return_original_text=False, root_data_dir=None, long_clip='disable'):
        # if root_data_dir is not None:
        #     df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.captions = df[caption_key].tolist()
        self.tokenize = tokenizer
        self.return_original_text = return_original_text
        self.context_length = 248 if long_clip != 'disable' else 77

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        original_text = str(self.captions[idx])
        texts = self.tokenize([original_text], context_length=self.context_length)[0]
        if self.return_original_text:
            return texts, original_text
        return texts


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def run(model_arch_name, pretrained, dataset_name, force_quick_gelu=False, long_clip=False):
    long_clip = 'load_from_scratch' if long_clip else 'disable'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    random_seed(42, 0)

    # Load model
    model, _, preprocess_val = create_model_and_transforms(
        model_arch_name,
        pretrained,
        precision=precision,
        device=device,
        output_dict=True,
        force_quick_gelu=force_quick_gelu,
        long_clip=long_clip
    )
    tokenizer = get_tokenizer(model_arch_name)
    model.eval()

    # Load data
    data_csv_path = DATA_CSV_PATH_DICT[dataset_name]
    if dataset_name == 'SkyScript':
        caption_key = 'title_multi_objects'
        ROOT_DATA_DIR = SKYSCRIPT_IMAGE_DIR
    else:
        caption_key = 'title'
        ROOT_DATA_DIR = RETRIEVAL_IMAGE_DIR

    df = pd.read_csv(data_csv_path)
    df['filepath'] = df['filepath'].apply(lambda x: join(ROOT_DATA_DIR, x))
    if dataset_name in ['RSICD', 'RSITMD', 'ucmcaptions']:
        df[caption_key] = df[caption_key].apply(lambda x: 'a satellite image. ' + x)

    df_image = df.groupby('filepath').count().reset_index()
    df_text = df.groupby(caption_key).count().reset_index()

    # Extract image features
    dataset_image = CsvDataset_image(
        df=df_image,
        transforms=preprocess_val,
        img_key='filepath',
        return_img_path=True,
    )
    dataloader = DataLoader(dataset_image, batch_size=batch_size, shuffle=False, num_workers=4)

    all_image_features = []
    all_image_paths = []
    with torch.no_grad():
        for batch in tqdm(dataloader, unit_scale=batch_size):
            images, img_paths = batch
            images = images.to(device=device)
            with autocast():
                image_features = model.encode_image(images, normalize=True)
                all_image_features.append(image_features.cpu())
                all_image_paths.extend(img_paths)
    all_image_features = torch.cat(all_image_features)

    # Extract text features
    dataset_text = CsvDataset_text(
        df=df_text,
        caption_key=caption_key,
        tokenizer=tokenizer,
        return_original_text=True,
        long_clip=long_clip
    )

    dataloader = DataLoader(dataset_text, batch_size=batch_size, shuffle=False, num_workers=4)

    all_text_features = []
    all_texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader, unit_scale=batch_size):
            texts, original_texts = batch
            texts = texts.to(device=device)
            with autocast():
                text_features = model.encode_text(texts, normalize=True)
                all_text_features.append(text_features.cpu())
                all_texts.extend(original_texts)
    all_text_features = torch.cat(all_text_features)

    text_indices = {x: i for i, x in enumerate(all_texts)}
    img_indices = {x: i for i, x in enumerate(all_image_paths)}

    # ground truth
    img_path2text = {}
    text2img_path = {}
    for i in tqdm(df.index):
        text = df.loc[i, caption_key]
        img_path = df.loc[i, 'filepath']
        text_id = text_indices[text]
        img_id = img_indices[img_path]
        if img_path not in img_path2text:
            img_path2text[img_path] = set()
        img_path2text[img_path].add(text_id)
        if text not in text2img_path:
            text2img_path[text] = set()
        text2img_path[text].add(img_id)

    res = {'text2img_R@' + str(k): 0 for k in [1, 5, 10, 100]}
    res.update({'img2text_R@' + str(k): 0 for k in [1, 5, 10, 100]})

    # text to image
    logit_scale = 100
    for i in tqdm(range(len(all_texts))):
        text_feature = all_text_features[i]
        logits = logit_scale * text_feature @ all_image_features.t()
        ranking = torch.argsort(logits, descending=True).cpu().numpy()
        for k in [1, 5, 10, 100]:
            intersec = set(ranking[:k]) & set(text2img_path[all_texts[i]])
            if intersec:
                res['text2img_R@' + str(k)] += 1
    for k in [1, 5, 10, 100]:
        res['text2img_R@' + str(k)] /= len(all_texts)
    res['text2img_mean'] = (res['text2img_R@1'] + res['text2img_R@5'] + res['text2img_R@10']) / 3

    # image to text
    logit_scale = 100
    for i in tqdm(range(len(all_image_paths))):
        image_feature = all_image_features[i]
        logits = logit_scale * image_feature @ all_text_features.t()
        ranking = torch.argsort(logits, descending=True).cpu().numpy()
        for k in [1, 5, 10, 100]:
            intersec = set(ranking[:k]) & img_path2text[all_image_paths[i]]
            if intersec:
                res['img2text_R@' + str(k)] += 1
    for k in [1, 5, 10, 100]:
        res['img2text_R@' + str(k)] /= len(all_image_paths)
    res['img2text_mean'] = (res['img2text_R@1'] + res['img2text_R@5'] + res['img2text_R@10']) / 3

    return(res)


def run_baseline(model_arch_name, model_name, pretrained, force_quick_gelu=False, long_clip=False):

    acc_dict = {}
    for dataset_name in ['RSICD', 'RSITMD', 'ucmcaptions', 'SkyScript']:
        try:
            res = run(
                model_arch_name=model_arch_name,
                pretrained=pretrained,
                dataset_name=dataset_name,
                force_quick_gelu=force_quick_gelu,
                long_clip=long_clip
            )
            acc_dict[dataset_name] = res

        except Exception as e:
            print(f"Evaluate Dataset {dataset_name} failed.")

    # Save results
    save_dir = f'./results_retrieval/{model_arch_name}/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f'retrieval.txt')

    log_dict = {}
    metric_accum = defaultdict(list)
    for dataset, metrics in acc_dict.items():
        for metric_name, value in metrics.items():
            log_dict[f"{dataset}/{metric_name}"] = value
            metric_accum[metric_name].append(value)

    with open(output_file, "a") as f:
        for k, v in log_dict.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--model-arch', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--use-long-clip', action='store_true')
    parser.add_argument('--force-quick-gelu', action='store_true')

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()

    run_baseline(
        model_arch_name=args.model_arch,
        model_name=args.model_name,
        pretrained=args.pretrained,
        long_clip=args.use_long_clip,
        force_quick_gelu=args.force_quick_gelu
    )
