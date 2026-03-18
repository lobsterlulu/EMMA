import os
import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
import matplotlib
from src_files.models.tresnet.tresnet import InplacABN_to_ABN
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os
import csv
from src.config import *
import re

# matplotlib.use('Agg')  # 已经在之前设置为Agg

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='/home1/lu-wei/repo/EMMA/classifier/ML_Decoder/models_zoo/tresnet_l_COCO__448_90_0.pth')
parser.add_argument('--pic-dir', type=str, default='./pics')  # 改为目录
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--image-size', type=int, default=448)
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=20)
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)
parser.add_argument('--txtname', type=str, default='person')
parser.add_argument('--erased_concept', type=str, default='airplane')
parser.add_argument('--mapping_concept', type=str, default='mapping_sky')
parser.add_argument('--erasing_method', type=str, default='mace')
parser.add_argument('--image_folder', type=str, default=None)
parser.add_argument('--csv_folder', type=str, default=None)
parser.add_argument('--base_image_path', type=str, 
                    default='/home1/lu-wei/repo/EMMA/results/MACE/saved_image/nsfw/generated_images/',
                    help='基础图片路径，用于构建完整的图片路径')


def main():
    print('Inference code on a single image')

    # parsing args
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args, load_head=True).cuda()
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)

    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')

    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    csv_folder = args.csv_folder
    
    # 如果指定了特定的concept，只处理该concept
    if args.erased_concept:
        concepts_to_process = [args.erased_concept]
    else:
        # 否则处理csv_folder中的所有CSV文件
        concepts_to_process = []
        for file in os.listdir(csv_folder):
            if file.endswith('.csv'):
                concept = file[:-4]  # 去掉.csv扩展名
                concepts_to_process.append(concept)
    
    for concept in concepts_to_process:
        print(f"\n处理概念: {concept}")
        
        # csv_filename = os.path.join(csv_folder, f"{concept}.csv")
        csv_filename = os.path.join(csv_folder, f"classification.csv")
        
        if not os.path.exists(csv_filename):
            print(f"警告: CSV文件不存在 {csv_filename}")
            continue
            
        # 读取CSV文件
        df = pd.read_csv(csv_filename)
        
        # 如果MLdecoder列不存在，则添加
        if "MLdecoder" not in df.columns:
            df["MLdecoder"] = ""
        
        print(f"CSV文件结构:")
        print(df.head())
        print(f"共有 {len(df)} 张图片需要处理")
        
        # 对每张图片进行推理
        for index, row in df.iterrows():
            image_name = row["image_name"]
            concept_name = row["concept"]
            evaluation_metric = row["evaluation_metric"]
            
            # 根据固定的路径结构构建图片完整路径
            # /home1/lu-wei/repo/EMMA/results/MACE/saved_image/nsfw/generated_images/{concept}/6_random/erased/{image_name}
            # pic_path = os.path.join(args.base_image_path, concept_name, evaluation_metric, "erased", image_name)
            # pic_path = os.path.join(args.base_image_path, concept_name, evaluation_metric, image_name)
            pic_path = os.path.join(args.image_folder, image_name)
            
            if not os.path.exists(pic_path):
                print(f"Warning: {pic_path} 不存在，跳过")
                df.at[index, "MLdecoder"] = "FILE_NOT_FOUND"
                continue

            print(f'loading image: {pic_path}')

            try:
                # 加载图片并进行推理
                im = Image.open(pic_path)
                im_resize = im.resize((args.image_size, args.image_size))
                np_img = np.array(im_resize, dtype=np.uint8)
                tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
                tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half()  # float16 inference
                output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
                np_output = output.cpu().detach().numpy()

                ## Top-k predictions
                idx_sort = np.argsort(-np_output)
                detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
                scores = np_output[idx_sort][: args.top_k]
                idx_th = scores > args.th
                detected_classes = detected_classes[idx_th]

                # 存入 CSV 的 MLdecoder 列
                df.at[index, "MLdecoder"] = ", ".join(detected_classes)

                print(f"Detected classes for {image_name}: {detected_classes}")
                
            except Exception as e:
                print(f"Error processing {pic_path}: {str(e)}")
                df.at[index, "MLdecoder"] = f"ERROR: {str(e)}"

        # 保存更新后的CSV文件
        df.to_csv(csv_filename, index=False)
        print(f"已更新 CSV 文件 {csv_filename}，所有推理结果已写入 MLdecoder 列。")
        print(f"处理完成概念: {concept}")


if __name__ == '__main__':
    main()