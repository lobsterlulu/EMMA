import fire
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
use_gpu = True  # 或根据环境变量判断是否可用 GPU

if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import fsspec
from main.paper_experiments.experiments import run_model_imagefolder
from argparse import Namespace


clip_model_name = 'ViT-L/14'
prompt_path = f'data/{clip_model_name.replace("/", "-")}/prompts.p'


def main_imagedataset(input_folder, output_folder, gpu=0):
    """main function"""
    gpu_id = 0 if use_gpu else -1
    args = Namespace(
        language_model='Clip_'+clip_model_name,
        model_type='sim',
        prompt_path=prompt_path,
        only_inappropriate=True,
        input_type='img',
        gpu=[gpu_id],
        # gpu=[gpu] if isinstance(gpu, int) else [int(gpu)],
    )
    run_model_imagefolder(args, input_folder, output_folder)


if __name__ == '__main__':
    fire.Fire(main_imagedataset)
