"""
Art Style Classifier (UnlearnDiffAtk / Diffusion-MU-Attack)
Usage: python run.py --image_dir /path/to/images [--model_path classifier/checkpoint-2800] [--output_csv results.csv]
"""
import os
import sys
import argparse
import glob
import csv
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))


def get_images(image_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
        files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description='Art Style Classifier')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory of images to classify')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to HuggingFace model checkpoint (default: classifier/checkpoint-2800)')
    parser.add_argument('--output_csv', type=str, default='artstyle_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Top-k art style predictions to record')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.join(SCRIPT_DIR, 'classifier', 'checkpoint-2800')

    device = args.gpu if args.gpu >= 0 else -1

    from utils.metrics.style_eval import init_classifier, style_eval
    print(f'Loading model from {args.model_path} on device {device}...')
    classifier = init_classifier(device, args.model_path)
    print('Model loaded.')

    image_files = get_images(args.image_dir)
    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Classifying {len(image_files)} images...')
    results_out = []
    for img_path in tqdm(image_files):
        try:
            image = Image.open(img_path).convert('RGB')
            preds = style_eval(classifier, image)[:args.top_k]
            top_labels = [p['label'] for p in preds]
            top_scores = [round(p['score'], 4) for p in preds]
            results_out.append({
                'image': os.path.basename(img_path),
                'top1_style': top_labels[0],
                'top1_score': top_scores[0],
                'top_predictions': ', '.join(
                    f'{l}({s:.4f})' for l, s in zip(top_labels, top_scores)),
            })
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            results_out.append({
                'image': os.path.basename(img_path),
                'top1_style': 'ERROR',
                'top1_score': 0.0,
                'top_predictions': str(e),
            })

    print(f'\n=== Results ===')
    print(f'Total images: {len(results_out)}')

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'top1_style', 'top1_score', 'top_predictions'])
        writer.writeheader()
        writer.writerows(results_out)
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
