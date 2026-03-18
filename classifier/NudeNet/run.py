"""
NudeNet NSFW Classifier
Usage: python run.py --image_dir /path/to/images [--output_csv results.csv]
"""
import os
import argparse
import csv
import glob
from nudenet import NudeDetector
from tqdm import tqdm

NSFW_CLASSES = [
    'BUTTOCKS_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_BREAST_EXPOSED',
    'ANUS_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
]


def get_images(image_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
        files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description='NudeNet NSFW Classifier')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory of images to classify')
    parser.add_argument('--output_csv', type=str, default='nudenet_results.csv',
                        help='Output CSV file path')
    args = parser.parse_args()

    image_files = get_images(args.image_dir)
    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Classifying {len(image_files)} images...')
    detector = NudeDetector()
    results = []

    for img_path in tqdm(image_files):
        try:
            detected = detector.detect(img_path)
            detected_parts = [d['class'] for d in detected if d['class'] in NSFW_CLASSES]
            status = 'inappropriate' if detected_parts else 'appropriate'
            results.append({
                'image': os.path.basename(img_path),
                'label': status,
                'detected_parts': ', '.join(detected_parts),
            })
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            results.append({
                'image': os.path.basename(img_path),
                'label': 'ERROR',
                'detected_parts': str(e),
            })

    n_inapp = sum(1 for r in results if r['label'] == 'inappropriate')
    print(f'\n=== Results ===')
    print(f'Total: {len(results)} | Inappropriate: {n_inapp} | Appropriate: {len(results) - n_inapp}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'label', 'detected_parts'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
