"""
Batch predict classification results for 30 logos (simplified version)
Predict on generated_images under each logo directory and aggregate results
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import re
import sys

# 30 target logos (matching directory names exactly)
LOGOS = [
    # Food
    'heineken', 'nestle', 'guinness', 'mcdonalds',
    # Clothes
    'asics', 'gap', 'converse', 'lacoste',
    # Necessities
    'colgate', 'nivea', 'gillette', 'pantene', 'neutrogena',
    # Electronic
    'apple', 'canon', 'asus', 'htc',
    # Transportation
    'bmw', 'lexus', 'lamborghini', 'chevrolet', 'michelin',
    # Leisure
    'marvel', 'barbie', 'hotwheels', 'play-doh',
    # Sports
    'spalding', 'oakley', 'underarmour', 'adidassb'
]

# Logo display name mapping (used for output and accuracy calculation)
LOGO_DISPLAY_NAMES = {
    'mcdonalds': "McDonald's",
    'hotwheels': 'Hot Wheels',
    'play-doh': 'play-doh',
    'underarmour': 'under armour',
    'adidassb': 'Adidas SB',
    'guinness': 'GUINNESS',
}


def normalize_name(name):
    """Normalize name for comparison"""
    return name.lower().replace("'", "").replace(" ", "").replace("-", "")


def find_logo_directories(base_dir):
    """Find all existing logo directories"""
    base_path = Path(base_dir)
    found_dirs = {}

    print("\nScanning logo directories:")
    print("-" * 60)

    for logo in LOGOS:
        logo_dir = base_path / logo / "generated_images"

        if logo_dir.exists() and logo_dir.is_dir():
            # Check if there are images
            image_files = list(logo_dir.glob('*.jpg')) + list(logo_dir.glob('*.png'))
            if len(image_files) > 0:
                found_dirs[logo] = logo_dir
                print(f"✓ {logo:20s} : {len(image_files)} images")
            else:
                print(f"✗ {logo:20s} : directory exists but no images")
        else:
            print(f"✗ {logo:20s} : directory does not exist")

    print("-" * 60)
    print(f"Found {len(found_dirs)}/{len(LOGOS)} valid directories\n")

    return found_dirs


def run_single_prediction(model_path, source_dir, logo_name):
    """
    Run prediction for a single logo

    Returns: (success, stdout, stderr)
    """
    cmd = [
        'python',
        '/home1/lu-wei/repo/EMMA/classifier/YOLO/src/predict_logo.py',
        '--model', str(model_path),
        '--source', str(source_dir),
        '--mode', 'batch'
    ]

    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  
            cwd='/home1/lu-wei/repo/EMMA/classifier/YOLO/src'  
        )

        return (result.returncode == 0, result.stdout, result.stderr)

    except subprocess.TimeoutExpired:
        return (False, "", "Prediction timed out (>600 seconds)")
    except Exception as e:
        return (False, "", str(e))


def parse_prediction_output(stdout, true_logo):
    """
    Parse prediction results from output

    Returns: dict with predictions and accuracy
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

    if predictions:
        true_label = LOGO_DISPLAY_NAMES.get(true_logo, true_logo)
        true_normalized = normalize_name(true_label)

        correct = 0
        for pred_info in predictions.values():
            pred_normalized = normalize_name(pred_info['prediction'])
            if pred_normalized == true_normalized:
                correct += 1

        total = len(predictions)
        accuracy = (correct / total) * 100 if total > 0 else 0

        return {
            'predictions': predictions,
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }

    return None


def save_results(all_results, output_dir):
    """Save results to CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_data = []

    for logo in LOGOS:
        if logo in all_results and all_results[logo]['success']:
            data = all_results[logo]
            summary_data.append({
                'logo': LOGO_DISPLAY_NAMES.get(logo, logo),
                'total_images': data['total'],
                'correct': data['correct'],
                'accuracy': f"{data['accuracy']:.2f}%"
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = output_path / "accuracy_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")


def print_summary(all_results):
    """Print summary"""
    print("\n" + "=" * 80)
    print("Batch Prediction Results Summary")
    print("=" * 80)
    print(f"\n{'Logo':<20s} {'Images':<10s} {'Correct':<10s} {'Accuracy':<15s} {'Status':<10s}")
    print("-" * 80)

    total_images = 0
    total_correct = 0
    successful = 0
    failed = 0

    for logo in LOGOS:
        display_name = LOGO_DISPLAY_NAMES.get(logo, logo)

        if logo in all_results:
            result = all_results[logo]

            if result['success']:
                print(f"{display_name:<20s} {result['total']:<10d} {result['correct']:<10d} "
                      f"{result['accuracy']:<14.2f}% {'✓ Success':<10s}")

                total_images += result['total']
                total_correct += result['correct']
                successful += 1
            else:
                error = result.get('error', 'Unknown error')[:30]
                print(f"{display_name:<20s} {'-':<10s} {'-':<10s} {'-':<15s} ✗ {error}")
                failed += 1
        else:
            print(f"{display_name:<20s} {'-':<10s} {'-':<10s} {'-':<15s} {'✗ Not run':<10s}")
            failed += 1

    print("-" * 80)
    overall_acc = (total_correct / total_images * 100) if total_images > 0 else 0
    print(f"{'Total':<20s} {total_images:<10d} {total_correct:<10d} {overall_acc:<14.2f}%")
    print(f"\nSucceeded: {successful}/{len(LOGOS)} | Failed: {failed}/{len(LOGOS)}")
    print("=" * 80)


def main():
    """Main function"""
    # Configuration
    model_path = "/home1/lu-wei/repo/EMMA/classifier/YOLO/runs/classify/logo_35_yolo11s2/weights/best.pt"
    base_dir = "/home1/lu-wei/repo/EMMA/results/sd21/saved_image/time_analysis/copyright"
    output_dir = "/home1/lu-wei/repo/EMMA/classifier/YOLO/batch_predictions"

    print("=" * 80)
    print("Batch Logo Prediction Tool (Simplified)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Image directory: {base_dir}")
    print(f"Output directory: {output_dir}")

    if not Path(model_path).exists():
        print(f"\n Error: Model file does not exist: {model_path}")
        return

    found_dirs = find_logo_directories(base_dir)

    if not found_dirs:
        print("\n No usable logo directories found!")
        return

    # Confirm
    response = input(f"\n Start prediction for these {len(found_dirs)} logos? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    # Start prediction
    print("\n" + "=" * 80)
    print("Starting batch prediction...")
    print("=" * 80)

    all_results = {}

    for i, (logo, source_dir) in enumerate(found_dirs.items(), 1):
        display_name = LOGO_DISPLAY_NAMES.get(logo, logo)

        print(f"\n[{i}/{len(found_dirs)}] Predicting: {display_name}")
        print("=" * 60)

        # Run prediction
        success, stdout, stderr = run_single_prediction(model_path, source_dir, logo)

        if success and stdout:
            # Parse results
            parsed = parse_prediction_output(stdout, logo)

            if parsed:
                all_results[logo] = {
                    'success': True,
                    'total': parsed['total'],
                    'correct': parsed['correct'],
                    'accuracy': parsed['accuracy']
                }
                print(f"\n Prediction complete: {parsed['correct']}/{parsed['total']} "
                      f"({parsed['accuracy']:.2f}%)")
            else:
                all_results[logo] = {
                    'success': False,
                    'error': 'Failed to parse output'
                }
                print(f"\n Failed: Could not parse prediction output")
                # Debug: print partial output
                print(f"\n First 500 characters of output:\n{stdout[:500]}")
        else:
            all_results[logo] = {
                'success': False,
                'error': stderr[:100] if stderr else 'Prediction command failed'
            }
            print(f"\n Prediction failed:")
            print(f"stderr: {stderr[:200]}")

    # Save and print results
    save_results(all_results, output_dir)
    print_summary(all_results)

    print(f"\n Batch prediction complete!")
    print(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
