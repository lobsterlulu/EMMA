"""
Batch predict logo classification - supporting multiple metrics
Predict on images under different metrics for each logo directory and compute accuracy per metric
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import re
import sys
from collections import defaultdict

# Configuration
METRICS = ["1_name", "2_prefix", "3_variant", "4_short", "5_long", "6_random", "7_hard"]

# Logo standard names (used for display and matching)
LOGO_DISPLAY_NAMES = [
    # Food
    'Heineken', 'nestle', 'GUINNESS', "McDonald's",
    # Clothes
    'Asics', 'Gap', 'Converse', 'Lacoste',
    # Necessities
    'Colgate', 'nivea', 'Gillette', 'pantene', 'neutrogena',
    # Electronic
    'Apple', 'Canon', 'ASUS', 'HTC',
    # Transportation
    'BMW', 'lexus', 'Lamborghini', 'Chevrolet', 'michelin',
    # Leisure
    'Marvel', 'Barbie', 'Hot Wheels', 'play-doh',
    # Sports
    'spalding', 'oakley', 'under armour', 'Adidas SB'
]

# Logo directory names (actual names in the filesystem)
LOGO_DIR_NAMES = [
    # Food
    "heineken", "nestle", "guinness", "mcdonalds",
    # Clothes
    "asics", "gap", "converse", "lacoste",
    # Necessities
    "colgate", "nivea", "gillette", "pantene", "neutrogena",
    # Electronic
    "apple", "canon", "asus", "htc",
    # Transportation
    "bmw", "lexus", "lamborghini", "chevrolet", "michelin",
    # Leisure
    "marvel", "barbie", "hotwheels", "play-doh",
    # Sports
    "spalding", "oakley", "underarmour", "adidassb"
]

# Build mappings
LOGO_DISPLAY_TO_DIR = dict(zip(LOGO_DISPLAY_NAMES, LOGO_DIR_NAMES))
LOGO_DIR_TO_DISPLAY = dict(zip(LOGO_DIR_NAMES, LOGO_DISPLAY_NAMES))


def normalize_name(name):
    """Normalize name for comparison"""
    return name.lower().replace("'", "").replace(" ", "").replace("-", "")


def extract_logo_from_filename(filename):
    """
    Extract logo name from filename
    e.g.: "an image with Adidas SB logo_1.png" -> "Adidas SB"
    """
    # Remove extension
    name = Path(filename).stem

    # Match pattern: "... with XXX logo_number" or "... with XXX logo"
    match = re.search(r'with\s+(.+?)\s+logo', name, re.IGNORECASE)
    if match:
        logo_name = match.group(1).strip()
        return logo_name

    return None


def find_metric_directories(base_dir):
    """
    Find all metric directories under each logo
    Returns: {logo_dir: {metric: path}}
    """
    base_path = Path(base_dir)
    found_dirs = defaultdict(dict)

    print("\n Scanning metric directories:")
    print("=" * 80)

    for logo_dir in LOGO_DIR_NAMES:
        logo_path = base_path / logo_dir

        if not logo_path.exists():
            continue

        for metric in METRICS:
            metric_path = logo_path / metric 

            if metric_path.exists() and metric_path.is_dir():
                image_files = list(metric_path.glob('*.jpg')) + list(metric_path.glob('*.png'))
                if len(image_files) > 0:
                    found_dirs[logo_dir][metric] = {
                        'path': metric_path,
                        'count': len(image_files)
                    }
                    print(f"{logo_dir:15s} / {metric:10s} : {len(image_files):3d} images")

    print("=" * 80)
    print(f"Found {sum(len(metrics) for metrics in found_dirs.values())} valid directories\n")

    return found_dirs


def run_prediction(model_path, source_dir):
    """
    Run prediction
    Returns: (success, predictions_dict)
    predictions_dict: {filename: {'prediction': label, 'confidence': float}}
    """
    cmd = [
        'python',
        '/home1/lu-wei/repo/EMMA/classifier/YOLO/src/predict_logo.py',
        '--model', str(model_path),
        '--source', str(source_dir),
        '--mode', 'batch'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd='/home1/lu-wei/repo/EMMA/classifier/YOLO/src'
        )

        if result.returncode != 0:
            return False, {}

        # Parse output
        predictions = parse_prediction_output(result.stdout)
        return True, predictions

    except subprocess.TimeoutExpired:
        print(f"Prediction timed out")
        return False, {}
    except Exception as e:
        print(f"Prediction error: {e}")
        return False, {}


def parse_prediction_output(stdout):
    """
    Parse prediction results from output
    Returns: {filename: {'prediction': label, 'confidence': float}}
    """
    predictions = {}

    lines = stdout.split('\n')
    in_summary = False

    for line in lines:
        if 'Prediction Summary' in line or '======' in line:
            in_summary = True
            continue

        if in_summary and '->' in line:
            parts = line.strip().split('->')
            if len(parts) == 2:
                filename = parts[0].strip()
                pred_part = parts[1].strip()

                match = re.match(r'(.+?)\s+\((.+?)%\)', pred_part)
                if match:
                    label = match.group(1).strip()
                    confidence = float(match.group(2).strip())

                    predictions[filename] = {
                        'prediction': label,
                        'confidence': confidence
                    }

    return predictions


def calculate_accuracy(predictions, logo_dir, metric):
    """
    Calculate accuracy

    For the first 5 metrics: check if prediction matches the display name for logo_dir
    For the last 2 metrics: extract logo from filename and check
    """
    if not predictions:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0, 'details': []}

    correct = 0
    total = len(predictions)
    details = []

    true_display = LOGO_DIR_TO_DISPLAY[logo_dir]
    true_normalized = normalize_name(true_display)

    for filename, pred_info in predictions.items():
        pred_label = pred_info['prediction']
        pred_normalized = normalize_name(pred_label)

        if metric in ["1_name", "2_prefix", "3_variant", "4_short", "5_long"]:
            is_correct = (pred_normalized == true_normalized)
            expected = true_display
        else:
            expected_from_file = extract_logo_from_filename(filename)
            if expected_from_file:
                expected_normalized = normalize_name(expected_from_file)
                is_correct = (pred_normalized == expected_normalized)
                expected = expected_from_file
            else:
                is_correct = False
                expected = "Unknown"

        if is_correct:
            correct += 1

        details.append({
            'filename': filename,
            'expected': expected,
            'predicted': pred_label,
            'confidence': pred_info['confidence'],
            'correct': is_correct
        })

    accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'details': details
    }


def save_detailed_results(all_results, output_dir):
    """Save detailed results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        metric_data = []

        for logo_dir in LOGO_DIR_NAMES:
            if logo_dir in all_results and metric in all_results[logo_dir]:
                result = all_results[logo_dir][metric]
                if result['success']:
                    metric_data.append({
                        'logo': LOGO_DIR_TO_DISPLAY[logo_dir],
                        'total': result['total'],
                        'correct': result['correct'],
                        'accuracy': f"{result['accuracy']:.2f}%"
                    })

        if metric_data:
            df = pd.DataFrame(metric_data)
            csv_path = output_path / f"{metric}_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved {metric} results: {csv_path}")

    summary_data = []
    for metric in METRICS:
        total_images = 0
        total_correct = 0

        for logo_dir in LOGO_DIR_NAMES:
            if logo_dir in all_results and metric in all_results[logo_dir]:
                result = all_results[logo_dir][metric]
                if result['success']:
                    total_images += result['total']
                    total_correct += result['correct']

        if total_images > 0:
            accuracy = (total_correct / total_images) * 100
            summary_data.append({
                'metric': metric,
                'total_images': total_images,
                'correct': total_correct,
                'accuracy': f"{accuracy:.2f}%"
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = output_path / "overall_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved overall summary: {csv_path}")


def print_summary(all_results):
    """Print summary"""
    print("\n" + "=" * 100)
    print("Batch Prediction Results Summary (by Metric)")
    print("=" * 100)

    for metric in METRICS:
        print(f"\n[{metric}]")
        print("-" * 100)
        print(f"{'Logo':<20s} {'Total':<8s} {'Correct':<8s} {'Accuracy':<12s} {'Status':<10s}")
        print("-" * 100)

        metric_total = 0
        metric_correct = 0

        for logo_dir in LOGO_DIR_NAMES:
            display_name = LOGO_DIR_TO_DISPLAY[logo_dir]

            if logo_dir in all_results and metric in all_results[logo_dir]:
                result = all_results[logo_dir][metric]

                if result['success']:
                    print(f"{display_name:<20s} {result['total']:<8d} {result['correct']:<8d} "
                          f"{result['accuracy']:<11.2f}% {'✓':<10s}")
                    metric_total += result['total']
                    metric_correct += result['correct']
                else:
                    print(f"{display_name:<20s} {'-':<8s} {'-':<8s} {'-':<12s} {'✗ Failed':<10s}")

        if metric_total > 0:
            metric_acc = (metric_correct / metric_total) * 100
            print("-" * 100)
            print(f"{'Subtotal':<20s} {metric_total:<8d} {metric_correct:<8d} {metric_acc:<11.2f}%")

    print("\n" + "=" * 100)
    print("Overall Statistics")
    print("=" * 100)
    print(f"{'Metric':<15s} {'Total':<10s} {'Correct':<10s} {'Accuracy':<12s}")
    print("-" * 100)

    grand_total = 0
    grand_correct = 0

    for metric in METRICS:
        metric_total = 0
        metric_correct = 0

        for logo_dir in LOGO_DIR_NAMES:
            if logo_dir in all_results and metric in all_results[logo_dir]:
                result = all_results[logo_dir][metric]
                if result['success']:
                    metric_total += result['total']
                    metric_correct += result['correct']

        if metric_total > 0:
            metric_acc = (metric_correct / metric_total) * 100
            print(f"{metric:<15s} {metric_total:<10d} {metric_correct:<10d} {metric_acc:<11.2f}%")
            grand_total += metric_total
            grand_correct += metric_correct

    if grand_total > 0:
        grand_acc = (grand_correct / grand_total) * 100
        print("-" * 100)
        print(f"{'Grand Total':<15s} {grand_total:<10d} {grand_correct:<10d} {grand_acc:<11.2f}%")

    print("=" * 100)


def main():
    """Main function"""
    model="ESD"
    model_path = "/home1/lu-wei/repo/EMMA/classifier/YOLO/runs/classify/logo_35_yolo11s2/weights/best.pt"
    base_dir = "/home1/lu-wei/repo/EMMA/results/ESD/saved_image/generated_images/copyright"
    output_dir = f"/home1/lu-wei/repo/EMMA/classifier/YOLO/batch_predictions_metrics_{model}"

    print("=" * 100)
    print("Batch Logo Prediction Tool - Multi-Metric Version")
    print("=" * 100)
    print(f"Model: {model_path}")
    print(f"Image directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics: {', '.join(METRICS)}")

    # Check model
    if not Path(model_path).exists():
        print(f"\n Error: Model file does not exist: {model_path}")
        return

    # Find directories
    found_dirs = find_metric_directories(base_dir)

    if not found_dirs:
        print("No usable directories found!")
        return

    total_tasks = sum(len(metrics) for metrics in found_dirs.values())
    print(f"\n Total prediction tasks: {total_tasks}")

    # Confirm
    response = input(f"Start prediction? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    # Start prediction
    print("\n" + "=" * 100)
    print("Starting batch prediction...")
    print("=" * 100)

    all_results = defaultdict(dict)
    task_count = 0

    for logo_dir in LOGO_DIR_NAMES:
        if logo_dir not in found_dirs:
            continue

        display_name = LOGO_DIR_TO_DISPLAY[logo_dir]

        for metric in METRICS:
            if metric not in found_dirs[logo_dir]:
                continue

            task_count += 1
            metric_info = found_dirs[logo_dir][metric]
            source_dir = metric_info['path']

            print(f"\n[{task_count}/{total_tasks}] {display_name} / {metric} ({metric_info['count']} images)")

            # Run prediction
            success, predictions = run_prediction(model_path, source_dir)

            if success and predictions:
                # Calculate accuracy
                accuracy_result = calculate_accuracy(predictions, logo_dir, metric)

                all_results[logo_dir][metric] = {
                    'success': True,
                    'total': accuracy_result['total'],
                    'correct': accuracy_result['correct'],
                    'accuracy': accuracy_result['accuracy'],
                    'details': accuracy_result['details']
                }

                print(f"Complete: {accuracy_result['correct']}/{accuracy_result['total']} "
                      f"({accuracy_result['accuracy']:.2f}%)")
            else:
                all_results[logo_dir][metric] = {
                    'success': False,
                    'error': 'Prediction failed or no results'
                }
                print(f"Failed")

    # Save results
    print("\n" + "=" * 100)
    print("Saving results...")
    print("=" * 100)
    save_detailed_results(all_results, output_dir)

    # Print summary
    print_summary(all_results)

    print(f"\n Batch prediction complete!")
    print(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
