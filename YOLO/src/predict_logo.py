"""
YOLOv11 Logo分类 - 推理脚本
使用训练好的模型进行logo识别
"""

from ultralytics import YOLO
import argparse
from pathlib import Path

def predict_single_image(model_path, image_path, top_k=5):
    """预测单张图片"""
    # 加载模型
    model = YOLO(model_path)
    
    # 预测
    results = model.predict(
        source=image_path,
        imgsz=224,
        device=0,  # GPU 0，使用CPU则改为'cpu'
        save=True,  # 保存预测结果
        conf=0.01,  # 置信度阈值
    )
    
    # 显示结果
    for result in results:
        probs = result.probs
        top_indices = probs.top5
        top_conf = probs.top5conf.tolist()
        
        print(f"\n图片: {image_path}")
        print(f"预测结果 (Top-{min(top_k, len(top_indices))}):")
        print("-" * 50)
        
        for i, (idx, conf) in enumerate(zip(top_indices[:top_k], top_conf[:top_k])):
            class_name = result.names[idx]
            print(f"{i+1}. {class_name:20s}: {conf*100:.2f}%")
        
        # 获取最高置信度的预测
        best_idx = top_indices[0]
        best_name = result.names[best_idx]
        best_conf = top_conf[0]
        
        print(f"\n最终预测: {best_name} (置信度: {best_conf*100:.2f}%)")
        
    return results


def predict_batch(model_path, image_dir, output_dir=None):
    """批量预测多张图片"""
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有图片
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    print(f"\n找到 {len(image_files)} 张图片")
    
    # 批量预测
    results = model.predict(
        source=str(image_dir),
        imgsz=224,
        device=0,
        save=True,
        conf=0.01,
    )
    
    # 汇总结果
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
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("预测结果汇总")
    print("=" * 70)
    
    for img_name, pred in predictions.items():
        print(f"{img_name:30s} -> {pred['class']:20s} ({pred['confidence']*100:.2f}%)")
    
    return predictions


def validate_on_testset(model_path, data_root):
    """在测试集上验证模型"""
    # 加载模型
    model = YOLO(model_path)
    
    # 验证
    metrics = model.val(
        data=data_root,
        split='test',
        imgsz=224,
        device=0,
    )
    
    print("\n" + "=" * 70)
    print("测试集验证结果")
    print("=" * 70)
    print(f"Top-1 准确率: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
    print(f"Top-5 准确率: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Logo分类推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径 (如: runs/classify/logo_35/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True, help='图片路径或图片目录')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch', 'val'], 
                        help='推理模式: single(单张), batch(批量), val(验证)')
    parser.add_argument('--data', type=str, default='/home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35_enhanced',
                        help='数据集根目录(仅val模式需要)')
    parser.add_argument('--top-k', type=int, default=5, help='显示Top-K结果')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        predict_single_image(args.model, args.source, args.top_k)
    elif args.mode == 'batch':
        predict_batch(args.model, args.source)
    elif args.mode == 'val':
        validate_on_testset(args.model, args.data)


if __name__ == '__main__':
    # 使用示例
    # 
    # 1. 预测单张图片:
    #    python predict.py --model runs/classify/logo_35/weights/best.pt --source test.jpg --mode single
    # 
    # 2. 批量预测:
    #    python predict.py --model /home1/lu-wei/repo/EMMA/classifier/YOLO/runs/classify/logo_35_yolo11s/weights/best.pt --source /home1/lu-wei/repo/EMMA/results/sd21/saved_image/time_analysis/copyright/adidassb/generated_images --mode batch
    # 
    # 3. 验证测试集:
    #    python predict.py --model runs/classify/logo_35/weights/best.pt --mode val
    
    main()
