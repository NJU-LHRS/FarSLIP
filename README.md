<h1 align="center"> FarSLIP: Discovering Effective CLIP Adaptation for Fine-Grained Remote Sensing Understanding </h1> 

<p align="center">
    <a href="">
        <img alt="Hugging Face Dataset" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue">
    </a>
    <a href="https://huggingface.co/datasets/ZhenShiL/MGRS-200k">
        <img alt="Hugging Face Model" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow">
    </a>
    <a href="https://huggingface.co/ZhenShiL/FarSLIP">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b">
    </a>
</p>


## Introduction
We introduce FarSLIP, a vision-language foundation model for remote sensing (RS) that achieves fine-grained vision-language alignment. FarSLIP demonstrates state-of-the-art performance on both fine-grained and image-level tasks, including open-vocabulary semantic segmentation, zero-shot classification, and image-text retrieval.
We also construct MGRS-200k, the first multi-granularity image-text dataset for RS. Each image is annotated with both short and long global-level captions, along with multiple object-category pairs.

<figure>
<div align="center">
<img src=assets/model.png width="50%">
</div>
</figure>


## Table of Contents
- [Introduction](#Introduction)
- [Preparation](#Preparation)
  - [Installation](#Installation)
  - [Checkpoints](#Checkpoints)
  - [Dataset](#Dataset)
- [Training](#Training)
- [Testing](#Testing)
    - [Open-vocabulary semantic segmentation](#open-vocabulary-semantic-segmentation)
    - [Zero-shot scene classification](#zero-shot-scene-classification)
    - [Zero-shot image-text retrieval](#zero-shot-image-text-retrieval)
- [Acknowledgement](#Acknowledgement)
- [Citing](#Citing)





## Preparation

### Installation

1. Clone this repository.

    ~~~shell
    git clone git@github.com:NJU-LHRS/FarSLIP.git
    cd FarSLIP
    ~~~

2. Create a new virtual environment

    ~~~shell
    conda create -n farslip python=3.10
    conda activate farslip
    ~~~

3. Install dependences

    ~~~shell
    pip install -r requirements.txt
    ~~~

### Checkpoints
You can download all our checkpoints from [Huggingface](https://huggingface.co/ZhenShiL/FarSLIP), or selectively download them through the links below.

| Model name  | Architecture | OVSS mIoU (%) | ZSC top-1 accuracy (%) | Download |
|-------------|--------------|---------------|-------------------------|----------------|
| FarSLIP-s1  | ViT-B-32     | 29.87         | 58.64                  | [FarSLIP1_ViT-B-32](https://huggingface.co/ZhenShiL/FarSLIP/resolve/main/FarSLIP1_ViT-B-32.pt?download=true) |
| FarSLIP-s2  | ViT-B-32     | 30.49         | 60.12                  | [FarSLIP2_ViT-B-32](https://huggingface.co/ZhenShiL/FarSLIP/resolve/main/FarSLIP2_ViT-B-32.pt?download=true) |
| FarSLIP-s1  | ViT-B-16     | 35.44         | 61.89                  | [FarSLIP1_ViT-B-16](https://huggingface.co/ZhenShiL/FarSLIP/resolve/main/FarSLIP1_ViT-B-16.pt?download=true) |
| FarSLIP-s2  | ViT-B-16     | 35.41         | 62.24                  | [FarSLIP2_ViT-B-16](https://huggingface.co/ZhenShiL/FarSLIP/resolve/main/FarSLIP2_ViT-B-16.pt?download=true) |


### Dataset
FarSLIP is trained in two stages.
+ In the first stage, we use the [RS5M](https://github.com/om-ai-lab/RS5M) dataset. A quick portal to the RS5M dataset: [link](https://huggingface.co/datasets/omlab/RS5M).
+ In the second stage, we use the proposed MGRS-200k dataset, which is available on [Huggingface](https://huggingface.co/datasets/ZhenShiL/MGRS-200k).

[//]: # (<figure>)

[//]: # (<div align="center">)

[//]: # (<img src=assets/dataset.png width="80%">)

[//]: # (</div>)

[//]: # (<figcaption align="center"><em>Examples from MGRS-200k</em></figcaption>)

[//]: # (</figure>)

<p align="center">
  <img src="assets/dataset.png" width="100%">
  <br>
  <em>Examples from MGRS-200k</em>
</p>

## Training

+ Validation data preparation
    + Replace --root-val-img-dir and --val-data in [config.py](./open_clip_train/config.py) with the paths to your SkyScript validation dataset.
+ Stage1
    ~~~shell
    torchrun --nproc_per_node=4 -m open_clip_train.main --train-dataset-name RS5M --train-data '/your/path/to/rs5m/{pub11,rs3}-train-{0000..0031}.tar' --train-dataset-type webdataset --train-num-samples 5070186 --method farslip1 --use-imagecrop-aug --local-method randomcrops --warmup 1000 --batch-size 10 --lr 1e-6 --wd 1.0 --epochs 1 --model ViT-B-16 --loss-type global_itc distill --distill-align roi2pooled
    ~~~

+ Stage2
    ~~~shell
    torchrun --nproc_per_node=4 -m open_clip_train.main --train-dataset-name MGRS --root-train-img-dir '/your/path/to/mgrs/global_imgs/' --train-data '/your/path/to/mgrs/text_info.json' --train-dataset-type json --method farslip2 --local-method randomcrops --warmup 250 --batch-size 10 --lr 4e-9 --wd 1.0 --epochs 10 --model ViT-B-16 --loss-type global_itc distill local_itc --distill-align roi2pooled --local-itc-align cls
    ~~~

## Testing
### Open-vocabulary semantic segmentation
+ Please checkout [FarSLIP-OVSS](https://github.com/NJU-LHRS/FarSLIP-OVSS) for evaluation of open-vocabulary semantic segmentation in RS images.

<p align="center">
  <img src="assets/ovss.png" width="100%">
  <br>
  <em>
    OVSS accuracies across RS benchmarks (mIoU, %). G denotes general-domain models, and RS refers to RS-specific models.
    f. indicates models specifically designed with fine-grained optimization. All models use an input image size of 224, except TIPS (448)
  </em>
</p>



### Zero-shot scene classification
+ Please refer to [SkyScript](https://github.com/wangzhecheng/SkyScript?tab=readme-ov-file) for scene classification dataset preparation, including 'SkyScript', 'aid', 'eurosat', 'fmow', 'millionaid', 'patternnet', 'rsicb', 'nwpu'.
+ Replace the BENCHMARK_DATASET_ROOT_DIR in [test_scene_classification.py](./tests/test_scene_classification.py) to your own path.

+ Run testing (e.g. FarSLIP-s1 with ViT-B-32):
```
python -m tests.test_scene_classification --model-arch ViT-B-32 --model-name FarSLIP1 --force-quick-gelu --pretrained checkpoints/FarSLIP1_ViT-B-32.pt
```


<figure>
<div align="center">
<img src=assets/classification.png width="100%">
</div>
<figcaption align="center">
<em>Comparison of zero-shot classification accuracies (Top-1 acc., %) of different RS-specific CLIP variants across multiple benchmarks.</em>
</figcaption>
</figure>


### Zero-shot image-text retrieval
+ Please refer to [SkyScript](https://github.com/wangzhecheng/SkyScript?tab=readme-ov-file) for image-text retrieval dataset preparation, including 'RSICD', 'RSITMD', 'ucmcaptions', 'SkyScript'.
+ Replace the data_csv_path_dict, SkyScript_dir, Retrieval_dir in [test_retrieval.py](./tests/test_retrieval.py) to your own path.

+ Run testing (e.g. FarSLIP-s1 with ViT-B-32):
```
python -m tests.test_retrieval --model-arch ViT-B-32 --model-name FarSLIP1 --force-quick-gelu --pretrained checkpoints/FarSLIP1_ViT-B-32.pt
```

<div align="center">
<img src=assets/retrieval.png width="50%">
</div>
<figcaption align="center">
<em>Comparison of cross-modal retrieval accuracies (%) of different RS-specific CLIP variants across multiple benchmarks. *
indicates models trained with in-hold supervision.</em>
</figcaption>
</figure>




## Acknowledgement

+ We gratitude to the following repositories for their wonderful works: [Open-CLIP](https://github.com/mlfoundations/open_clip), [CLIPSelf](https://github.com/wusize/CLIPSelf), [FineCLIP](https://github.com/Timsty1/FineCLIP), [Long-CLIP](https://github.com/beichenzbc/Long-CLIP), [SkyScript](https://github.com/wangzhecheng/SkyScript), [SegEarth](https://github.com/likyoo/SegEarth-OV).


## Citing

+ If you find our work is useful, please give us 🌟 in GitHub and consider cite our paper:

    ~~~tex

    ~~~
