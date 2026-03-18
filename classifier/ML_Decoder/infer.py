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
parser.add_argument('--model-path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
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

    # 读取 CSV 文件
    # print(" args.erasing_method: ",  args.erasing_method)
    # if args.erasing_method == "mace":
    #     csv_folder = os.path.join(MACE_SAVE_RESULTS_CSV_DIR, "image_info")
    # elif args.erasing_method == "uce":
    #     csv_folder = os.path.join(UCE_SAVE_RESULTS_CSV_DIR, "image_info")
    # elif args.erasing_method == "esd":
    #     csv_folder = os.path.join(ESD_SAVE_RESULTS_CSV_DIR, "image_info")
    # else:
    #     print("Error: No exsting methods!")
    # # print("csv_folder: ",csv_folder)
    
    csv_folder = args.csv_folder
    
    csv_filename = os.path.join(csv_folder, f"{args.erased_concept}.csv")

    df = pd.read_csv(csv_filename)
    df["MLdecoder"] = ""


    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}  # 可根据需要扩展

    def natural_sort_key(file_path):
        """提取文件名中的数字部分并进行自然排序"""
        filename = os.path.basename(file_path)
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
    
    # if args.erasing_method == "mace":
    #     # /home1/lu-wei/repo/MACE/save_results/object/sd14_generated/airplane/mapping_sky/efficacy_01/erased
    #     # image_folder = os.path.join(MACE_SAVE_RESULTS_DIR, f"sd14_generated/{args.erased_concept}/mapping_sky") # sd14
    #     image_folder = f"/data/lu-wei/repo/MACE/save_results/object/object/sd21_generated/{args.erased_concept}/mapping_sky"
    #     # image_folder = os.path.join(MACE_SAVE_RESULTS_DIR, f"generated_images/{args.erased_concept}/mapping_sky") 
    # elif args.erasing_method == "uce":
    #     image_folder = os.path.join(UCE_SAVE_RESULTS_DIR, f"{args.erased_concept}/hard_specificity") 
    # elif args.erasing_method == "esd":
    #     image_folder = os.path.join(ESD_SAVE_RESULTS_DIR, f"{args.erased_concept}/hard_specificity") 
    # else:
    #     print("Error: No exsting methods!")

    image_folder = args.image_folder
    all_images = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_images.append(os.path.join(root, file))

    # 按文件名进行自然排序
    all_images.sort(key=natural_sort_key)

    csv_data = [["index", "image_name", "erased_object", "evaluation_metric"]]
    for idx, img_path in enumerate(all_images, start=1):
        img_name = os.path.basename(img_path)
        path_parts = img_path.split(os.sep)
        
        # print("img_name:", img_name)
        image_folder_absolute = os.path.abspath(image_folder)  # 转换为绝对路径
        parent_folder = os.path.basename(os.path.dirname(image_folder_absolute))
        original_folder = os.path.basename(image_folder)
        sub_folder = path_parts[len(image_folder.split(os.sep))] if len(path_parts) > len(image_folder.split(os.sep)) else "N/A"
        
        new_img_name = f"{img_name}"
        csv_data.append([idx, new_img_name, parent_folder, sub_folder])

    # with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    #     writer = csv.writer(file)
    #     writer.writerows(csv_data)

    print(f"已记录 {len(all_images)} 张 PNG 图片的新名称至 {csv_filename}。")
    df = pd.read_csv(csv_filename)
    print(df.head())

    
    # 对每张图片进行推理
    for index, row in df.iterrows():
        image_name = row["image_name"]
        evaluation_metric = row["evaluation_metric"]
        if args.erasing_method == "mace":
            pic_path = os.path.join(image_folder,evaluation_metric,"erased",image_name)
        elif args.erasing_method == "uce" or args.erasing_method == "esd":
            # pic_path = os.path.join(image_folder,evaluation_metric,image_name)
            pic_path = os.path.join(image_folder,image_name) #uce&esd
        elif args.erasing_method == "ca":
            pic_path = os.path.join(image_folder,evaluation_metric,"samples",image_name)
        elif args.erasing_method == "fmn":
            pic_path = os.path.join(image_folder,evaluation_metric,image_name)
        else:
            print("Error: No exsting methods!")

        if not os.path.exists(pic_path):
            print(f"Warning: {pic_path} 不存在，跳过")
            continue

        print(f'loading image: {pic_path}')

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

    # 直接保存 CSV，替换原文件
    df.to_csv(csv_filename, index=False)
    print(f"已更新 CSV 文件 {csv_filename}，所有推理结果已写入 MLdecoder 列。")

if __name__ == '__main__':
    main()
