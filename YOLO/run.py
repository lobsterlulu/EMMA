"""
YOLO Logo Classifier
Usage: python run.py --image_dir /path/to/images [--model_path model/best.pt] [--output_csv results.csv]
"""
import os
import sys
import argparse
import glob
import csv
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_images(image_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
        files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description='YOLO Logo Classifier')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory of images to classify')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to YOLO .pt weights (default: largest file in model/)')
    parser.add_argument('--output_csv', type=str, default='yolo_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Top-k predictions to record per image')
    parser.add_argument('--conf', type=float, default=0.01,
                        help='Confidence threshold')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (-1 for CPU)')
    args = parser.parse_args()

    if args.model_path is None:
        model_dir = Path(os.path.join(SCRIPT_DIR, 'model'))
        candidates = list(model_dir.glob('*.pt'))
        if not candidates:
            print(f'No .pt model found in {model_dir}. Use --model_path to specify one.')
            return
        # Prefer yolo11x-cls (largest/most accurate) if present
        preferred = [p for p in candidates if 'yolo11x' in p.name]
        args.model_path = str(preferred[0] if preferred else sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0])
        print(f'Using model: {args.model_path}')

    from ultralytics import YOLO
    model = YOLO(args.model_path)
    device = args.gpu if args.gpu >= 0 else 'cpu'

    image_files = get_images(args.image_dir)
    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Classifying {len(image_files)} images...')
    results_out = []
    for img_path in tqdm(image_files):
        try:
            results = model.predict(source=img_path, imgsz=224, device=device,
                                    conf=args.conf, verbose=False)
            for result in results:
                probs = result.probs
                top_indices = probs.top5[:args.top_k]
                top_conf = probs.top5conf.tolist()[:args.top_k]
                top_labels = [result.names[i] for i in top_indices]
                results_out.append({
                    'image': os.path.basename(img_path),
                    'top1_label': top_labels[0],
                    'top1_conf': round(top_conf[0], 4),
                    'top_predictions': ', '.join(
                        f'{l}({c:.4f})' for l, c in zip(top_labels, top_conf)),
                })
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            results_out.append({
                'image': os.path.basename(img_path),
                'top1_label': 'ERROR',
                'top1_conf': 0.0,
                'top_predictions': str(e),
            })

    print(f'\n=== Results ===')
    print(f'Total images: {len(results_out)}')

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'top1_label', 'top1_conf', 'top_predictions'])
        writer.writeheader()
        writer.writerows(results_out)
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
