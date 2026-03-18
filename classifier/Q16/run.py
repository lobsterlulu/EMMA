"""
Q16 NSFW Classifier
Usage: python run.py --image_dir /path/to/images [--output_csv results.csv] [--model ViT-L/14] [--gpu 0]
"""
import os
import sys
import argparse
import glob
import csv
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from classify.utils import ClipWrapper, SimClassifier, load_prompts

LABELS = ['non inappropriate', 'inappropriate']


def get_images(image_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
        files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description='Q16 NSFW Classifier')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory of images to classify')
    parser.add_argument('--output_csv', type=str, default='q16_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--model', type=str, default='ViT-L/14',
                        choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14'],
                        help='CLIP model variant')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    model_key = args.model.replace('/', '-')
    prompt_path = os.path.join(SCRIPT_DIR, 'data', model_key, 'prompts.p')

    print(f'Loading CLIP {args.model} on {device}...')
    clip_wrapper = ClipWrapper(device, model_name=args.model)
    preprocess = clip_wrapper.preprocess

    print(f'Loading prompts from {prompt_path}...')
    prompts = load_prompts(prompt_path, device)
    classifier = SimClassifier(prompts, device).to(device)
    classifier.eval()

    image_files = get_images(args.image_dir)
    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Classifying {len(image_files)} images...')
    results = []
    with torch.no_grad():
        for img_path in tqdm(image_files):
            try:
                image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
                features = clip_wrapper(image)
                logits = classifier(features)
                probs = logits.softmax(dim=-1)
                pred_idx = probs.argmax().item()
                label = LABELS[pred_idx]
                score = probs[pred_idx].item()
                results.append({
                    'image': os.path.basename(img_path),
                    'label': label,
                    'score': round(score, 4),
                })
            except Exception as e:
                print(f'Error processing {img_path}: {e}')
                results.append({'image': os.path.basename(img_path), 'label': 'ERROR', 'score': 0.0})

    n_inapp = sum(1 for r in results if r['label'] == 'inappropriate')
    print(f'\n=== Results ===')
    print(f'Total: {len(results)} | Inappropriate: {n_inapp} | Appropriate: {len(results) - n_inapp}')

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'label', 'score'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
