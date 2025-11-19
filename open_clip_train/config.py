BENCHMARK_DATASET_ROOT_DIR = '/path/to/classification/benchmarks'     # benchmark_dataset_info root of scene classification datasets

arg_dict = {

    # '--train-dataset-name': 'MGRS',
    # '--root-train-img-dir': '.../MGRS/global_imgs',
    # '--train-data': '.../MGRS/text_info.json',
    # '--train-dataset-type': 'json',
    # '--method': 'farslip2',
    # '--local-method': 'randomcrops',

    # '--train-dataset-name': 'RS5M',
    # '--train-data': '.../rs5m/{pub11,rs3}-train-{0000..0031}.tar',
    # '--train-dataset-type': 'webdataset',
    # '--train-num-samples': 5070186,
    # '--method': 'farslip1',
    # "--use-imagecrop-aug": None,
    # '--local-method': 'randomcrops',

    '--root-val-img-dir': '.../skyscript/',
    '--val-data': '.../skyscript/SkyScript_val_5K_filtered_by_CLIP_openai.csv',
    '--val-dataset-type': 'csv',
    '--csv-img-key': 'filepath',
    '--csv-class-key': 'label',
    '--csv-caption-key': 'title',
    '--csv-separator': ',',

    '--warmup': 500,
    '--batch-size': 40,
    '--lr': 1e-6,
    '--wd': 1.0,
    '--epochs': 10,
    '--workers': 8,

    '--model': 'ViT-B-16',
    '--pretrained': 'openai',
    '--report-to': 'wandb',
    '--log-every-n-steps': 50,

    '--loss-type':
        [
            "local_itc",
            "global_itc",
            "distill"
        ],
    '--max-boxes': 4,
    '--max-size': 1024,
    '--min-size': 64,
    '--EMA-momentum': 0.99,
    '--force-quick-gelu': None,
    "--local-itc-align": 'cls',
    "--distill-align": 'roi2pooled',
    "--last-attn-type": "SegEarth",
    '--wandb-tags': [],
    '--long-clip': 'disable',           # "disable", "load_from_clip", "load_from_scratch"
    "--distill-type": "active",
    "--mpcl-loss": None,
    '--save-frequency': 1,
    # "--find-unused-parameters": None,
}