# YOLOv11 Logo分类训练指南

## 📁 项目结构

```
EMMA/classifier/YOLO/
├── logo_dataset_35_enhanced/          # 数据集
│   ├── train/                         # 训练集
│   │   ├── Apple/
│   │   ├── BMW/
│   │   ├── Coca-Cola/
│   │   └── ...                        # 35个品牌
│   └── test/                          # 测试集
│       ├── Apple/
│       ├── BMW/
│       └── ...
│
├── runs/classify/                     # 训练输出
│   └── logo_35_xxx/                   # 每次训练的结果
│       ├── weights/
│       │   ├── best.pt               # 最佳模型
│       │   └── last.pt               # 最后一次训练
│       ├── results.png               # 训练曲线
│       ├── confusion_matrix.png      # 混淆矩阵
│       └── ...
│
└── src/                               # 代码
    ├── train_yolo_classifier.py      # 完整训练脚本
    ├── train_simple.py               # 简化训练脚本
    ├── predict_logo.py               # 推理脚本
    └── prepare_yolo_dataset_enhanced.py  # 数据准备脚本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装ultralytics
pip install ultralytics --break-system-packages

# 验证安装
python -c "from ultralytics import YOLO; print('✓ Ultralytics installed')"
```

### 2. 训练模型

#### 方式一：使用简化脚本（推荐快速测试）

```bash
cd /home1/lu-wei/repo/EMMA/classifier/YOLO
python src/train_simple.py
```

#### 方式二：使用完整脚本（推荐正式训练）

```bash
cd /home1/lu-wei/repo/EMMA/classifier/YOLO
python src/train_yolo_classifier.py
```

#### 方式三：命令行（最灵活）

```bash
yolo classify train \
    data=/home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35_enhanced \
    model=yolo11n-cls.pt \
    epochs=100 \
    imgsz=224 \
    batch=32 \
    device=0 \
    project=/home1/lu-wei/repo/EMMA/classifier/YOLO/runs/classify \
    name=logo_35_cli
```

### 3. 监控训练

训练过程中会实时显示：
- Epoch进度
- Loss值
- 准确率
- 学习率

训练完成后查看结果：
```bash
cd runs/classify/logo_35_xxx
ls -lh weights/  # 查看模型文件
```

### 4. 验证模型

```bash
# Python方式
python src/predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --mode val

# 或命令行方式
yolo classify val \
    model=runs/classify/logo_35_xxx/weights/best.pt \
    data=/home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35_enhanced \
    imgsz=224
```

### 5. 推理预测

#### 单张图片预测

```bash
python src/predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source test.jpg \
    --mode single \
    --top-k 5
```

#### 批量预测

```bash
python src/predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source test_images/ \
    --mode batch
```

## 📊 模型选择

根据你的需求选择不同大小的模型：

| 模型 | 准确率 | 速度 | 参数量 | 推荐场景 |
|------|--------|------|--------|----------|
| yolo11n-cls | 70% | 最快 | 2.8M | 快速测试 |
| yolo11s-cls | 75% | 快 | 6.7M | 平衡选择 |
| yolo11m-cls | 77% | 中等 | 11.6M | 高精度 |
| yolo11l-cls | 78% | 慢 | 14.1M | 最高精度 |
| yolo11x-cls | 79% | 最慢 | 29.6M | 极致精度 |

**建议**：
- 35个类别，数据量3500+张 → 建议使用 **yolo11s-cls** 或 **yolo11m-cls**
- 如果训练时间充足 → 使用 **yolo11m-cls**
- 如果需要快速验证 → 使用 **yolo11n-cls**

## ⚙️ 重要参数说明

### 训练参数

```python
epochs=100          # 训练轮数，可根据收敛情况调整
imgsz=224          # 图像大小，标准为224
batch=32           # 批次大小，根据显存调整
                   # 显存12GB: batch=32
                   # 显存8GB: batch=16
                   # 显存4GB: batch=8

patience=20        # 早停耐心值，20轮不提升则停止
lr0=0.01          # 初始学习率
lrf=0.01          # 最终学习率因子
```

### 数据增强参数

```python
fliplr=0.5        # 水平翻转概率
hsv_h=0.015       # 色调增强
hsv_s=0.7         # 饱和度增强
hsv_v=0.4         # 明度增强
degrees=10        # 随机旋转角度
scale=0.5         # 随机缩放范围
```

## 🔍 查看训练结果

训练完成后，在 `runs/classify/logo_35_xxx/` 目录下：

```bash
# 查看训练曲线
display results.png

# 查看混淆矩阵
display confusion_matrix.png

# 查看验证结果
cat results.csv
```

## 📈 性能优化建议

### 1. 如果准确率不高
- 增加训练轮数 (epochs=150 或 200)
- 使用更大的模型 (yolo11m-cls 或 yolo11l-cls)
- 调整学习率 (lr0=0.001 尝试更小的学习率)
- 增加数据增强强度

### 2. 如果过拟合
- 增加patience值
- 增加数据增强
- 使用更小的模型
- 减小学习率

### 3. 如果训练太慢
- 减小batch size
- 使用更小的模型 (yolo11n-cls)
- 使用混合精度训练 (amp=True)

## 🎯 预期结果

基于35个类别，3500+张图片：
- **Top-1 准确率**: 70-85%
- **Top-5 准确率**: 90-95%
- **训练时间**: 
  - yolo11n-cls: ~30-45分钟 (100 epochs)
  - yolo11s-cls: ~45-60分钟
  - yolo11m-cls: ~60-90分钟

## 🐛 常见问题

### Q1: CUDA out of memory
```bash
# 解决方案：减小batch size
batch=16  # 或 batch=8
```

### Q2: 训练速度很慢
```bash
# 检查是否使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 如果False，设置device
device=0  # 使用GPU 0
```

### Q3: 找不到数据集
```bash
# 确认数据集路径正确
ls -la /home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35_enhanced/train
ls -la /home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35_enhanced/test
```

## 📝 导出模型

训练完成后，可以导出为ONNX等格式用于部署：

```python
from ultralytics import YOLO

model = YOLO('runs/classify/logo_35_xxx/weights/best.pt')
model.export(format='onnx')  # 导出为ONNX
```

## 📚 参考资料

- [Ultralytics YOLOv11文档](https://docs.ultralytics.com/)
- [分类任务文档](https://docs.ultralytics.com/tasks/classify/)
- [训练配置参数](https://docs.ultralytics.com/modes/train/)
