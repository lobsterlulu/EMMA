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

# matplotlib.use('Agg')  # Already set to Agg previously

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='/home1/lu-wei/repo/EMMA/classifier/ML_Decoder/models_zoo/tresnet_l_COCO__448_90_0.pth')
parser.add_argument('--pic-dir', type=str, default='./pics')  # Changed to directory
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
                    help='Base image path for constructing full image paths')


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
    
    # If a specific concept is specified, only process that concept
    if args.erased_concept:
        concepts_to_process = [args.erased_concept]
    else:
        # Otherwise process all CSV files in csv_folder
        concepts_to_process = []
        for file in os.listdir(csv_folder):
            if file.endswith('.csv'):
                concept = file[:-4]  # Remove .csv extension
                concepts_to_process.append(concept)
    
    for concept in concepts_to_process:
        print(f"\nProcessing concept: {concept}")
        
        # csv_filename = os.path.join(csv_folder, f"{concept}.csv")
        csv_filename = os.path.join(csv_folder, f"classification.csv")
        
        if not os.path.exists(csv_filename):
            print(f"Warning: CSV file does not exist {csv_filename}")
            continue
            
        # Read CSV file
        df = pd.read_csv(csv_filename)

        # Add MLdecoder column if it doesn't exist
        if "MLdecoder" not in df.columns:
            df["MLdecoder"] = ""

        print(f"CSV file structure:")
        print(df.head())
        print(f"Total {len(df)} images to process")
        
        # Run inference on each image
        for index, row in df.iterrows():
            image_name = row["image_name"]
            concept_name = row["concept"]
            evaluation_metric = row["evaluation_metric"]
            
            # Build full image path based on fixed path structure
            # /home1/lu-wei/repo/EMMA/results/MACE/saved_image/nsfw/generated_images/{concept}/6_random/erased/{image_name}
            # pic_path = os.path.join(args.base_image_path, concept_name, evaluation_metric, "erased", image_name)
            # pic_path = os.path.join(args.base_image_path, concept_name, evaluation_metric, image_name)
            pic_path = os.path.join(args.image_folder, image_name)
            
            if not os.path.exists(pic_path):
                print(f"Warning: {pic_path} does not exist, skipping")
                df.at[index, "MLdecoder"] = "FILE_NOT_FOUND"
                continue

            print(f'loading image: {pic_path}')

            try:
                # Load image and run inference
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

                # Store in the MLdecoder column of CSV
                df.at[index, "MLdecoder"] = ", ".join(detected_classes)

                print(f"Detected classes for {image_name}: {detected_classes}")

            except Exception as e:
                print(f"Error processing {pic_path}: {str(e)}")
                df.at[index, "MLdecoder"] = f"ERROR: {str(e)}"

        # Save updated CSV file
        df.to_csv(csv_filename, index=False)
        print(f"Updated CSV file {csv_filename}, all inference results written to MLdecoder column.")
        print(f"Finished processing concept: {concept}")


if __name__ == '__main__':
    main()