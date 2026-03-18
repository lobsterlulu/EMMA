"""
GCD (Giphy Celebrity Detector) - Celebrity Face Classifier
Usage: python run.py --image_dir /path/to/images [--output_csv results.csv] [--use_gpu]
"""
import os
import sys
import argparse
import glob
import csv
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point APP_DATA_DIR at the bundled resources before importing model_training
os.environ['APP_DATA_DIR'] = os.path.join(SCRIPT_DIR, 'resources') + os.sep

sys.path.insert(0, SCRIPT_DIR)

from skimage import io
from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.preprocessors.face_detection.face_detector import FaceDetector


def get_images(image_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
        files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description='GCD Celebrity Classifier')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory of images to classify')
    parser.add_argument('--output_csv', type=str, default='gcd_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k celebrity predictions to record')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for inference')
    args = parser.parse_args()

    resources_path = os.path.join(SCRIPT_DIR, 'resources') + os.sep
    image_size = 224

    print('Loading face detector and recognizer...')
    model_labels = Labels(resources_path=resources_path)
    face_detector = FaceDetector(resources_path, margin=0.2, use_cuda=args.use_gpu)
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=resources_path,
        use_cuda=args.use_gpu,
        top_n=args.top_k,
    )
    print('Models loaded.')

    image_files = get_images(args.image_dir)
    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Classifying {len(image_files)} images...')
    results_out = []

    for img_path in tqdm(image_files):
        try:
            image = io.imread(img_path)
            face_images = face_detector.perform_single(image)
            if not face_images:
                results_out.append({
                    'image': os.path.basename(img_path),
                    'face_detected': False,
                    'top1_celebrity': 'N/A',
                    'top1_score': 0.0,
                    'top_predictions': '',
                })
                continue

            face_images_proc = [preprocess_image(img, image_size) for img, _ in face_images]
            predictions = face_recognizer.perform(face_images_proc)

            if not predictions or not predictions[0]:
                results_out.append({
                    'image': os.path.basename(img_path),
                    'face_detected': True,
                    'top1_celebrity': 'N/A',
                    'top1_score': 0.0,
                    'top_predictions': '',
                })
                continue

            top_preds = predictions[0][0][:args.top_k]
            top_names = [str(label).split('_[', 1)[0].replace('_', ' ') for label, _ in top_preds]
            top_scores = [round(float(score), 4) for _, score in top_preds]
            results_out.append({
                'image': os.path.basename(img_path),
                'face_detected': True,
                'top1_celebrity': top_names[0],
                'top1_score': top_scores[0],
                'top_predictions': ', '.join(
                    f'{n}({s:.4f})' for n, s in zip(top_names, top_scores)),
            })
        except Exception as e:
            print(f'Error processing {img_path}: {e}')
            results_out.append({
                'image': os.path.basename(img_path),
                'face_detected': False,
                'top1_celebrity': 'ERROR',
                'top1_score': 0.0,
                'top_predictions': str(e),
            })

    n_faces = sum(1 for r in results_out if r['face_detected'])
    print(f'\n=== Results ===')
    print(f'Total: {len(results_out)} | Faces detected: {n_faces}')

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['image', 'face_detected', 'top1_celebrity', 'top1_score', 'top_predictions'])
        writer.writeheader()
        writer.writerows(results_out)
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
