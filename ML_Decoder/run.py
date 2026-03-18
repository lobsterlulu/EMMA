"""
ML-Decoder Classifier
Usage: python run.py --image_dir /path/to/images [--task nsfw|object] [--model_path /path/to/model.pth] [--output_csv results.csv]

Tasks:
  nsfw   - flags images containing exposed body parts (uses COCO-trained model)
  object - detects COCO 80-class objects present in each image
"""
import os
import sys
import argparse
import glob
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
from src_files.models.tresnet.tresnet import InplacABN_to_ABN

# Body-part classes treated as NSFW
NSFW_CLASSES = {
    'MALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_EXPOSED',
    'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
}

DEFAULT_MODEL = os.path.join(SCRIPT_DIR, 'models_zoo', 'tresnet_l_COCO__448_90_0.pth')


def get_images(image_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
        files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(set(files))


def load_model(model_path):
    from argparse import Namespace
    args = Namespace(
        model_name='tresnet_l',
        num_classes=80,
        use_ml_decoder=1,
        num_of_groups=-1,
        decoder_embedding=768,
        zsl=0,
    )
    model = create_model(args, load_head=True).cuda()
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    return model, classes_list


def classify_image(model, img_path, classes_list, image_size=448, th=0.75, top_k=20):
    im = Image.open(img_path).convert('RGB')
    im_resize = im.resize((image_size, image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half()
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)
    detected = np.array(classes_list)[idx_sort][:top_k]
    scores = np_output[idx_sort][:top_k]
    detected = detected[scores > th]
    return detected.tolist()


def main():
    parser = argparse.ArgumentParser(description='ML-Decoder Classifier')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory of images to classify')
    parser.add_argument('--task', type=str, default='object',
                        choices=['nsfw', 'object'],
                        help='Classification task (default: object)')
    parser.add_argument('--model_path', type=str, default=None,
                        help=f'Path to model .pth file (default: {DEFAULT_MODEL})')
    parser.add_argument('--output_csv', type=str, default='ml_decoder_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--th', type=float, default=0.75,
                        help='Detection confidence threshold')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Top-k classes to evaluate per image')
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = DEFAULT_MODEL

    print(f'Loading model from {args.model_path}...')
    model, classes_list = load_model(args.model_path)
    print('Model loaded.')

    image_files = get_images(args.image_dir)
    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Classifying {len(image_files)} images (task={args.task})...')
    results = []
    with torch.no_grad():
        for img_path in tqdm(image_files):
            try:
                detected = classify_image(model, img_path, classes_list, th=args.th, top_k=args.top_k)
                if args.task == 'nsfw':
                    nsfw_hits = [c for c in detected if c in NSFW_CLASSES]
                    label = 'inappropriate' if nsfw_hits else 'appropriate'
                    results.append({
                        'image': os.path.basename(img_path),
                        'label': label,
                        'detected_classes': ', '.join(detected),
                    })
                else:
                    results.append({
                        'image': os.path.basename(img_path),
                        'detected_classes': ', '.join(detected),
                    })
            except Exception as e:
                print(f'Error processing {img_path}: {e}')
                row = {'image': os.path.basename(img_path), 'detected_classes': f'ERROR: {e}'}
                if args.task == 'nsfw':
                    row['label'] = 'ERROR'
                results.append(row)

    print(f'\n=== Results ===')
    print(f'Total images: {len(results)}')
    if args.task == 'nsfw':
        n_inapp = sum(1 for r in results if r.get('label') == 'inappropriate')
        print(f'Inappropriate: {n_inapp} | Appropriate: {len(results) - n_inapp}')

    fields = (['image', 'label', 'detected_classes'] if args.task == 'nsfw'
              else ['image', 'detected_classes'])
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
