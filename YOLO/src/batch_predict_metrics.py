"""
批量预测Logo分类 - 支持多个metric
对每个logo目录下不同metric的图片进行预测，并按metric统计准确率
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import re
import sys
from collections import defaultdict

# 配置
METRICS = ["1_name", "2_prefix", "3_variant", "4_short", "5_long", "6_random", "7_hard"]

# Logo标准名称（用于显示和匹配）
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

# Logo目录名（实际文件系统中的名称）
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

# 建立映射关系
LOGO_DISPLAY_TO_DIR = dict(zip(LOGO_DISPLAY_NAMES, LOGO_DIR_NAMES))
LOGO_DIR_TO_DISPLAY = dict(zip(LOGO_DIR_NAMES, LOGO_DISPLAY_NAMES))


def normalize_name(name):
    """标准化名称用于比较"""
    return name.lower().replace("'", "").replace(" ", "").replace("-", "")


def extract_logo_from_filename(filename):
    """
    从文件名中提取logo名称
    例如: "an image with Adidas SB logo_1.png" -> "Adidas SB"
    """
    # 移除扩展名
    name = Path(filename).stem
    
    # 匹配模式: "... with XXX logo_数字" 或 "... with XXX logo"
    match = re.search(r'with\s+(.+?)\s+logo', name, re.IGNORECASE)
    if match:
        logo_name = match.group(1).strip()
        return logo_name
    
    return None


def find_metric_directories(base_dir):
    """
    查找所有logo下的metric目录
    返回: {logo_dir: {metric: path}}
    """
    base_path = Path(base_dir)
    found_dirs = defaultdict(dict)
    
    print("\n扫描metric目录:")
    print("=" * 80)
    
    for logo_dir in LOGO_DIR_NAMES:
        logo_path = base_path / logo_dir
        
        if not logo_path.exists():
            continue
        
        for metric in METRICS:
            metric_path = logo_path / metric #/ "erased"
            
            if metric_path.exists() and metric_path.is_dir():
                # 检查是否有图片
                image_files = list(metric_path.glob('*.jpg')) + list(metric_path.glob('*.png'))
                if len(image_files) > 0:
                    found_dirs[logo_dir][metric] = {
                        'path': metric_path,
                        'count': len(image_files)
                    }
                    print(f"✓ {logo_dir:15s} / {metric:10s} : {len(image_files):3d} 张图片")
    
    print("=" * 80)
    print(f"找到 {sum(len(metrics) for metrics in found_dirs.values())} 个有效目录\n")
    
    return found_dirs


def run_prediction(model_path, source_dir):
    """
    运行预测
    返回: (success, predictions_dict)
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
        
        # 解析输出
        predictions = parse_prediction_output(result.stdout)
        return True, predictions
    
    except subprocess.TimeoutExpired:
        print(f"  ⚠ 预测超时")
        return False, {}
    except Exception as e:
        print(f"  ⚠ 预测出错: {e}")
        return False, {}


def parse_prediction_output(stdout):
    """
    从输出中解析预测结果
    返回: {filename: {'prediction': label, 'confidence': float}}
    """
    predictions = {}
    
    lines = stdout.split('\n')
    in_summary = False
    
    for line in lines:
        if '预测结果汇总' in line or '======' in line:
            in_summary = True
            continue
        
        if in_summary and '->' in line:
            parts = line.strip().split('->')
            if len(parts) == 2:
                filename = parts[0].strip()
                pred_part = parts[1].strip()
                
                # 提取label和confidence
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
    计算准确率
    
    对于前5个metric: 判断预测是否等于logo_dir对应的显示名
    对于后2个metric: 从文件名提取logo并判断
    """
    if not predictions:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0, 'details': []}
    
    correct = 0
    total = len(predictions)
    details = []
    
    # 获取该logo的标准显示名
    true_display = LOGO_DIR_TO_DISPLAY[logo_dir]
    true_normalized = normalize_name(true_display)
    
    for filename, pred_info in predictions.items():
        pred_label = pred_info['prediction']
        pred_normalized = normalize_name(pred_label)
        
        # 判断依据
        if metric in ["1_name", "2_prefix", "3_variant", "4_short", "5_long"]:
            # 前5个metric: 根据目录名判断
            is_correct = (pred_normalized == true_normalized)
            expected = true_display
        else:
            # 后2个metric: 根据文件名判断
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
    """保存详细结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存每个metric的汇总
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
            print(f"  ✓ 保存 {metric} 结果: {csv_path}")
    
    # 2. 保存总体汇总
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
        print(f"  ✓ 保存总体汇总: {csv_path}")


def print_summary(all_results):
    """打印汇总"""
    print("\n" + "=" * 100)
    print("批量预测结果汇总 (按Metric)")
    print("=" * 100)
    
    for metric in METRICS:
        print(f"\n【{metric}】")
        print("-" * 100)
        print(f"{'Logo':<20s} {'总数':<8s} {'正确':<8s} {'准确率':<12s} {'状态':<10s}")
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
                    print(f"{display_name:<20s} {'-':<8s} {'-':<8s} {'-':<12s} {'✗ 失败':<10s}")
        
        if metric_total > 0:
            metric_acc = (metric_correct / metric_total) * 100
            print("-" * 100)
            print(f"{'小计':<20s} {metric_total:<8d} {metric_correct:<8d} {metric_acc:<11.2f}%")
    
    # 总体统计
    print("\n" + "=" * 100)
    print("总体统计")
    print("=" * 100)
    print(f"{'Metric':<15s} {'总数':<10s} {'正确':<10s} {'准确率':<12s}")
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
        print(f"{'总计':<15s} {grand_total:<10d} {grand_correct:<10d} {grand_acc:<11.2f}%")
    
    print("=" * 100)


def main():
    """主函数"""
    # 配置路径
    model="ESD"
    model_path = "/home1/lu-wei/repo/EMMA/classifier/YOLO/runs/classify/logo_35_yolo11s2/weights/best.pt"
    base_dir = "/home1/lu-wei/repo/EMMA/results/ESD/saved_image/generated_images/copyright"
    output_dir = f"/home1/lu-wei/repo/EMMA/classifier/YOLO/batch_predictions_metrics_{model}"
    
    print("=" * 100)
    print("批量Logo预测工具 - 多Metric版本")
    print("=" * 100)
    print(f"模型: {model_path}")
    print(f"图片目录: {base_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Metrics: {', '.join(METRICS)}")
    
    # 检查模型
    if not Path(model_path).exists():
        print(f"\n❌ 错误: 模型文件不存在: {model_path}")
        return
    
    # 查找目录
    found_dirs = find_metric_directories(base_dir)
    
    if not found_dirs:
        print("❌ 没有找到任何可用的目录!")
        return
    
    total_tasks = sum(len(metrics) for metrics in found_dirs.values())
    print(f"\n共有 {total_tasks} 个预测任务")
    
    # 确认
    response = input(f"是否开始预测? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 开始预测
    print("\n" + "=" * 100)
    print("开始批量预测...")
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
            
            print(f"\n[{task_count}/{total_tasks}] {display_name} / {metric} ({metric_info['count']} 张)")
            
            # 运行预测
            success, predictions = run_prediction(model_path, source_dir)
            
            if success and predictions:
                # 计算准确率
                accuracy_result = calculate_accuracy(predictions, logo_dir, metric)
                
                all_results[logo_dir][metric] = {
                    'success': True,
                    'total': accuracy_result['total'],
                    'correct': accuracy_result['correct'],
                    'accuracy': accuracy_result['accuracy'],
                    'details': accuracy_result['details']
                }
                
                print(f"  ✓ 完成: {accuracy_result['correct']}/{accuracy_result['total']} "
                      f"({accuracy_result['accuracy']:.2f}%)")
            else:
                all_results[logo_dir][metric] = {
                    'success': False,
                    'error': '预测失败或无结果'
                }
                print(f"  ✗ 失败")
    
    # 保存结果
    print("\n" + "=" * 100)
    print("保存结果...")
    print("=" * 100)
    save_detailed_results(all_results, output_dir)
    
    # 打印汇总
    print_summary(all_results)
    
    print(f"\n✅ 批量预测完成!")
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()