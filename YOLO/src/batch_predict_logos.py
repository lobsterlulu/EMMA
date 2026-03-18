"""
批量预测30个Logo的分类结果 (简化版)
对每个logo目录下的generated_images进行预测，并汇总结果
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import re
import sys

# 30个目标logo (与目录名完全一致)
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

# Logo显示名称映射（用于输出和准确率计算）
LOGO_DISPLAY_NAMES = {
    'mcdonalds': "McDonald's",
    'hotwheels': 'Hot Wheels',
    'play-doh': 'play-doh',
    'underarmour': 'under armour',
    'adidassb': 'Adidas SB',
    'guinness': 'GUINNESS',
}


def normalize_name(name):
    """标准化名称用于比较"""
    return name.lower().replace("'", "").replace(" ", "").replace("-", "")


def find_logo_directories(base_dir):
    """查找所有存在的logo目录"""
    base_path = Path(base_dir)
    found_dirs = {}
    
    print("\n扫描logo目录:")
    print("-" * 60)
    
    for logo in LOGOS:
        logo_dir = base_path / logo / "generated_images"
        
        if logo_dir.exists() and logo_dir.is_dir():
            # 检查是否有图片
            image_files = list(logo_dir.glob('*.jpg')) + list(logo_dir.glob('*.png'))
            if len(image_files) > 0:
                found_dirs[logo] = logo_dir
                print(f"✓ {logo:20s} : {len(image_files)} 张图片")
            else:
                print(f"✗ {logo:20s} : 目录存在但无图片")
        else:
            print(f"✗ {logo:20s} : 目录不存在")
    
    print("-" * 60)
    print(f"找到 {len(found_dirs)}/{len(LOGOS)} 个有效目录\n")
    
    return found_dirs


def run_single_prediction(model_path, source_dir, logo_name):
    """
    运行单个logo的预测
    
    返回: (success, stdout, stderr)
    """
    # 构建命令 - 直接使用你之前成功的命令格式
    cmd = [
        'python',
        '/home1/lu-wei/repo/EMMA/classifier/YOLO/src/predict_logo.py',
        '--model', str(model_path),
        '--source', str(source_dir),
        '--mode', 'batch'
    ]
    
    print(f"\n运行命令:")
    print(" ".join(cmd))
    print("-" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            cwd='/home1/lu-wei/repo/EMMA/classifier/YOLO/src'  # 设置工作目录
        )
        
        return (result.returncode == 0, result.stdout, result.stderr)
    
    except subprocess.TimeoutExpired:
        return (False, "", f"预测超时 (>600秒)")
    except Exception as e:
        return (False, "", str(e))


def parse_prediction_output(stdout, true_logo):
    """
    从输出中解析预测结果
    
    返回: dict with predictions and accuracy
    """
    predictions = {}
    
    # 找到预测结果汇总部分
    lines = stdout.split('\n')
    in_summary = False
    
    for line in lines:
        # 检测汇总开始
        if '预测结果汇总' in line or '======' in line:
            in_summary = True
            continue
        
        if in_summary and '->' in line:
            # 解析: filename -> label (confidence%)
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
    
    # 计算准确率
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
    """保存结果到CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 汇总结果
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
        print(f"\n✓ 结果已保存到: {csv_path}")


def print_summary(all_results):
    """打印汇总"""
    print("\n" + "=" * 80)
    print("批量预测结果汇总")
    print("=" * 80)
    print(f"\n{'Logo':<20s} {'图片数':<10s} {'正确数':<10s} {'准确率':<15s} {'状态':<10s}")
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
                      f"{result['accuracy']:<14.2f}% {'✓ 成功':<10s}")
                
                total_images += result['total']
                total_correct += result['correct']
                successful += 1
            else:
                error = result.get('error', '未知错误')[:30]
                print(f"{display_name:<20s} {'-':<10s} {'-':<10s} {'-':<15s} ✗ {error}")
                failed += 1
        else:
            print(f"{display_name:<20s} {'-':<10s} {'-':<10s} {'-':<15s} {'✗ 未运行':<10s}")
            failed += 1
    
    print("-" * 80)
    overall_acc = (total_correct / total_images * 100) if total_images > 0 else 0
    print(f"{'总计':<20s} {total_images:<10d} {total_correct:<10d} {overall_acc:<14.2f}%")
    print(f"\n成功: {successful}/{len(LOGOS)} | 失败: {failed}/{len(LOGOS)}")
    print("=" * 80)


def main():
    """主函数"""
    # 配置
    model_path = "/home1/lu-wei/repo/EMMA/classifier/YOLO/runs/classify/logo_35_yolo11s2/weights/best.pt"
    base_dir = "/home1/lu-wei/repo/EMMA/results/sd21/saved_image/time_analysis/copyright"
    output_dir = "/home1/lu-wei/repo/EMMA/classifier/YOLO/batch_predictions"
    
    print("=" * 80)
    print("批量Logo预测工具 (简化版)")
    print("=" * 80)
    print(f"模型: {model_path}")
    print(f"图片目录: {base_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查模型是否存在
    if not Path(model_path).exists():
        print(f"\n❌ 错误: 模型文件不存在: {model_path}")
        return
    
    # 查找logo目录
    found_dirs = find_logo_directories(base_dir)
    
    if not found_dirs:
        print("❌ 没有找到任何可用的logo目录!")
        return
    
    # 确认
    response = input(f"是否开始预测这 {len(found_dirs)} 个logo? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 开始预测
    print("\n" + "=" * 80)
    print("开始批量预测...")
    print("=" * 80)
    
    all_results = {}
    
    for i, (logo, source_dir) in enumerate(found_dirs.items(), 1):
        display_name = LOGO_DISPLAY_NAMES.get(logo, logo)
        
        print(f"\n[{i}/{len(found_dirs)}] 正在预测: {display_name}")
        print("=" * 60)
        
        # 运行预测
        success, stdout, stderr = run_single_prediction(model_path, source_dir, logo)
        
        if success and stdout:
            # 解析结果
            parsed = parse_prediction_output(stdout, logo)
            
            if parsed:
                all_results[logo] = {
                    'success': True,
                    'total': parsed['total'],
                    'correct': parsed['correct'],
                    'accuracy': parsed['accuracy']
                }
                print(f"\n✓ 预测完成: {parsed['correct']}/{parsed['total']} "
                      f"({parsed['accuracy']:.2f}%)")
            else:
                all_results[logo] = {
                    'success': False,
                    'error': '无法解析输出'
                }
                print(f"\n✗ 失败: 无法解析预测输出")
                # 调试：打印部分输出
                print(f"\n输出前500字符:\n{stdout[:500]}")
        else:
            all_results[logo] = {
                'success': False,
                'error': stderr[:100] if stderr else '预测命令失败'
            }
            print(f"\n✗ 预测失败:")
            print(f"stderr: {stderr[:200]}")
    
    # 保存和打印结果
    save_results(all_results, output_dir)
    print_summary(all_results)
    
    print(f"\n✅ 批量预测完成!")
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()