"""
YOLOv11 Logo Classification - Inference Script
Use a trained model for logo recognition
"""

from ultralytics import YOLO
import argparse
from pathlib import Path

def predict_single_image(model_path, image_path, top_k=5):
    """Predict a single image"""
    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        imgsz=224,
        device=0,  
        save=True,  
        conf=0.01,  
    )

    for result in results:
        probs = result.probs
        top_indices = probs.top5
        top_conf = probs.top5conf.tolist()

        print(f"\nImage: {image_path}")
        print(f"Prediction Results (Top-{min(top_k, len(top_indices))}):")
        print("-" * 50)

        for i, (idx, conf) in enumerate(zip(top_indices[:top_k], top_conf[:top_k])):
            class_name = result.names[idx]
            print(f"{i+1}. {class_name:20s}: {conf*100:.2f}%")

        best_idx = top_indices[0]
        best_name = result.names[best_idx]
        best_conf = top_conf[0]

        print(f"\nFinal Prediction: {best_name} (Confidence: {best_conf*100:.2f}%)")

    return results


def predict_batch(model_path, image_dir, output_dir=None):
    """Batch predict multiple images"""
    model = YOLO(model_path)

    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

    print(f"\nFound {len(image_files)} images")

    results = model.predict(
        source=str(image_dir),
        imgsz=224,
        device=0,
        save=True,
        conf=0.01,
    )

    predictions = {}
    for result in results:
        img_path = Path(result.path).name
        probs = result.probs
        best_idx = probs.top1
        best_name = result.names[best_idx]
        best_conf = probs.top1conf.item()

        predictions[img_path] = {
            'class': best_name,
            'confidence': best_conf
        }

    print("\n" + "=" * 70)
    print("Prediction Summary")
    print("=" * 70)

    for img_name, pred in predictions.items():
        print(f"{img_name:30s} -> {pred['class']:20s} ({pred['confidence']*100:.2f}%)")

    return predictions


def validate_on_testset(model_path, data_root):
    """Validate model on the test set"""
    model = YOLO(model_path)

    metrics = model.val(
        data=data_root,
        split='test',
        imgsz=224,
        device=0,
    )

    print("\n" + "=" * 70)
    print("Test Set Validation Results")
    print("=" * 70)
    print(f"Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
    print(f"Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Logo Classification Inference')
    parser.add_argument('--model', type=str, required=True, help='Model path (e.g.: runs/classify/logo_35/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True, help='Image path or image directory')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch', 'val'],
                        help='Inference mode: single (single image), batch (batch), val (validation)')
    parser.add_argument('--data', type=str, default='/home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35_enhanced',
                        help='Dataset root directory (only needed for val mode)')
    parser.add_argument('--top-k', type=int, default=5, help='Show Top-K results')

    args = parser.parse_args()

    if args.mode == 'single':
        predict_single_image(args.model, args.source, args.top_k)
    elif args.mode == 'batch':
        predict_batch(args.model, args.source)
    elif args.mode == 'val':
        validate_on_testset(args.model, args.data)


if __name__ == '__main__':
    # Usage examples
    #
    # 1. Predict a single image:
    #    python predict.py --model runs/classify/logo_35/weights/best.pt --source test.jpg --mode single
    #
    # 2. Batch prediction:
    #    python predict.py --model /home1/lu-wei/repo/EMMA/classifier/YOLO/runs/classify/logo_35_yolo11s/weights/best.pt --source /home1/lu-wei/repo/EMMA/results/sd21/saved_image/time_analysis/copyright/adidassb/generated_images --mode batch
    #
    # 3. Validate on test set:
    #    python predict.py --model runs/classify/logo_35/weights/best.pt --mode val

    main()
