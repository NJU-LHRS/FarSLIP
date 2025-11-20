from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from open_clip_train.test_zero_shot_classification import *
from open_clip_train.benchmark_dataset_info import BENCHMARK_DATASET_INFOMATION, BENCHMARK_DATASET_ROOT_DIR

from prettytable import PrettyTable


def single_model_table(model_name: str, metrics: dict, dataset_names: list):
    """
    Print a table of top-1/top-5 accuracies for a model.
    """

    table = PrettyTable()
    table.field_names = ["Model", "Metric"] + dataset_names + ["mean"]

    for metric_type in ["top1", "top5"]:
        row = [model_name, metric_type]
        for dataset in dataset_names:
            row.append(metrics.get(f"{dataset}-{metric_type}", "N/A"))
        row.append(metrics.get(f"mean-{metric_type}", "N/A"))
        table.add_row(row)

    return table


def run_baseline(model_arch, model_name, dataset_list, pretrained=None, force_quick_gelu=False, use_long_clip=False):

    zero_shot_metrics = {}
    for dataset in dataset_list:
        print(f"Running {dataset}")

        test_data = BENCHMARK_DATASET_INFOMATION[dataset]['test_data']
        classnames = BENCHMARK_DATASET_INFOMATION[dataset]['classnames']
        arg_list = [
            '--test-data-dir=' + BENCHMARK_DATASET_ROOT_DIR,
            '--classification-mode=multiclass',
            '--csv-separator=,',
            '--csv-img-key', 'filepath',
            '--csv-class-key', 'label',
            '--batch-size=128',
            '--workers=8',
            '--model=' + model_arch,
            '--pretrained=' + pretrained,
            '--test-data=' + test_data,
            '--classnames=' + classnames,
            '--test-data-name=' + dataset,
        ]
        if force_quick_gelu:
            arg_list.append('--force-quick-gelu')
        if use_long_clip:
            arg_list.append('--long-clip=load_from_scratch')

        results = test(arg_list)

        # Record accuracy
        for k, v in results.items():
            if type(v) in [float, int, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]:
                zero_shot_metrics[k] = v

    mean_top1 = sum(value for key, value in zero_shot_metrics.items() if 'top1' in key) / len(dataset_list)
    mean_top5 = sum(value for key, value in zero_shot_metrics.items() if 'top5' in key) / len(dataset_list)
    zero_shot_metrics.update({'mean-top1': mean_top1, 'mean-top5': mean_top5})

    save_dir = f'./results_classification/{model_arch}/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'zs_cls.txt')

    table = single_model_table(model_name, zero_shot_metrics, dataset_list)
    print(table)
    with open(save_path, "w") as f:
        f.write(str(table))


import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--model-arch', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--use-long-clip', action='store_true')
    parser.add_argument('--force-quick-gelu', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    dataset_list = [
        'SkyScript_cls',
        'aid',
        'eurosat',
        'fmow',
        'millionaid',
        'patternnet',
        'rsicb',
        'nwpu',
    ]

    args = parse_args()
    model_arch = args.model_arch
    model_name = args.model_name

    # Setting pretrained model path
    # if model_name == 'CLIP': pretrained = 'openai'
    # if model_arch == 'ViT-B-32':
    #     if model_name == 'FarSLIP1':
    #         pretrained = "checkpoints/FarSLIP1-ViT-B-32",
    #     elif model_name == 'FarSLIP2':
    #         pretrained = "checkpoints/FarSLIP2_ViT-B-32",
    #     elif model_name == 'RemoteCLIP':
    #         pretrained = '.../models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-L-14.pt'  # RemoteCLIP
    #     elif model_name == 'SkyCLIP':
    #         pretrained = '.../checkpoints/SkyCLIP_ViT_L14_top50pct/epoch_20.pt'  # SkyScript-L14
    #     elif model_name == 'GeoRSCLIP':
    #         pretrained = '.../checkpoints/GeoRSCLIP-ckpt/RS5M_ViT-L-14.pt'
    # elif model_arch =='ViT-B-16':
    #     if model_name == 'LRSCLIP':
    #         pretrained = '.../LRSCLIP_ViT-B-16.pt'
    #     elif model_name == 'FarSLIP1':
    #         pretrained = "checkpoints/FarSLIP1-ViT-B-16"
    #     elif model_name == 'FarSLIP2':
    #         pretrained = "checkpoints/FarSLIP2_ViT-B-16",

    run_baseline(
        model_arch=model_arch,
        model_name=model_name,
        dataset_list=dataset_list,
        pretrained=args.pretrained,
        use_long_clip=args.use_long_clip,
        force_quick_gelu=args.force_quick_gelu
    )
