# YOLOv11 Logo分类完整流程

## 📦 已提供的文件

1. **prepare_yolo_dataset_enhanced.py** - 数据集准备脚本（已运行）
2. **train_yolo_classifier.py** - 完整训练脚本
3. **train_simple.py** - 简化训练脚本
4. **train_logo_oneclick.sh** - 一键训练脚本（推荐）
5. **predict_logo.py** - 推理预测脚本
6. **check_environment.py** - 环境检查脚本
7. **README_YOLO_TRAINING.md** - 详细使用文档

## 🚀 快速开始（3步走）

### 第1步：检查环境

```bash
cd /home1/lu-wei/repo/EMMA/classifier/YOLO
python check_environment.py
```

如果显示"所有检查通过"，继续下一步。

### 第2步：开始训练

```bash
# 方式1：一键训练（最简单）
./train_logo_oneclick.sh

# 方式2：Python脚本
python train_simple.py

# 方式3：命令行
yolo classify train \
    data=logo_dataset_35_enhanced \
    model=yolo11s-cls.pt \
    epochs=100 \
    imgsz=224 \
    batch=32 \
    device=0
```

### 第3步：验证和预测

```bash
# 验证模型
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --mode val

# 预测单张图片
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source test.jpg \
    --mode single
```

## 📊 训练配置建议

### 根据显存选择配置

| 显存 | 模型 | Batch Size | 预计时间 |
|------|------|------------|----------|
| 4GB | yolo11n-cls | 8-16 | 60-90分钟 |
| 8GB | yolo11s-cls | 16-32 | 45-60分钟 |
| 12GB | yolo11m-cls | 32-64 | 60-90分钟 |
| 24GB | yolo11l-cls | 64-128 | 90-120分钟 |

**推荐配置（你的情况）**：
- 模型：**yolo11s-cls** 或 **yolo11m-cls**
- Batch size：**32**
- Epochs：**100**
- 预计Top-1准确率：**75-85%**

## 📁 目录结构

```
/home1/lu-wei/repo/EMMA/classifier/YOLO/
│
├── logo_dataset_35_enhanced/       ← 数据集（已准备好）
│   ├── train/                      ← 训练集：2849张图片
│   │   ├── Apple/
│   │   ├── BMW/
│   │   └── ... (35个品牌)
│   └── test/                       ← 测试集：729张图片
│       ├── Apple/
│       └── ...
│
├── src/                            ← 代码文件
│   ├── prepare_yolo_dataset_enhanced.py
│   ├── train_yolo_classifier.py
│   ├── train_simple.py
│   ├── train_logo_oneclick.sh
│   ├── predict_logo.py
│   └── check_environment.py
│
└── runs/classify/                  ← 训练输出（训练后生成）
    └── logo_35_xxx/
        ├── weights/
        │   ├── best.pt            ← 最佳模型
        │   └── last.pt            ← 最后模型
        ├── results.png            ← 训练曲线
        ├── confusion_matrix.png   ← 混淆矩阵
        └── results.csv            ← 详细结果
```

## 🎯 关键参数说明

```python
# 基础参数
model = 'yolo11s-cls.pt'    # 模型大小
epochs = 100                 # 训练轮数
imgsz = 224                  # 图像大小（标准）
batch = 32                   # 批次大小
device = 0                   # GPU编号

# 学习率
lr0 = 0.01                   # 初始学习率
lrf = 0.01                   # 最终学习率因子

# 早停
patience = 20                # 20轮不提升则停止

# 数据增强
fliplr = 0.5                 # 水平翻转
hsv_h = 0.015               # 色调
hsv_s = 0.7                 # 饱和度
hsv_v = 0.4                 # 明度
degrees = 10                 # 旋转角度
scale = 0.5                  # 缩放
```

## 📈 监控训练

训练过程中实时显示：

```
Epoch    GPU_mem   loss  top1_acc  top5_acc
  1/100     2.5G   3.52     0.15      0.45
  2/100     2.5G   2.98     0.28      0.62
  3/100     2.5G   2.45     0.42      0.75
  ...
100/100     2.5G   0.85     0.82      0.96
```

关键指标：
- **loss**: 越小越好
- **top1_acc**: Top-1准确率（主要指标）
- **top5_acc**: Top-5准确率

## 🔍 评估结果

训练完成后，查看结果：

```bash
cd runs/classify/logo_35_xxx

# 查看最终指标
cat results.csv | tail -5

# 查看混淆矩阵
display confusion_matrix.png

# 查看训练曲线
display results.png

# 查看模型大小
ls -lh weights/
```

## 🐛 常见问题

### 1. CUDA out of memory
```bash
# 减小batch size
batch=16  # 或 8
```

### 2. 训练速度慢
```bash
# 确认使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 使用更小的模型
model=yolo11n-cls.pt
```

### 3. 准确率不高
- 增加训练轮数：epochs=150
- 使用更大模型：yolo11m-cls.pt
- 检查数据质量和标注

### 4. 过拟合
- 增加数据增强
- 减小模型大小
- 早停：patience=20

## 📝 训练后的操作

### 导出模型
```python
from ultralytics import YOLO
model = YOLO('runs/classify/logo_35_xxx/weights/best.pt')
model.export(format='onnx')  # 导出为ONNX
```

### 在新图片上测试
```bash
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source new_logo.jpg \
    --mode single
```

### 批量推理
```bash
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source test_images/ \
    --mode batch
```

## 💡 优化建议

### 如果数据不平衡
某些类别图片太少（如Logitech只有7张）：

1. **数据增强**：增加augment参数
2. **类别权重**：使用weighted loss
3. **重采样**：oversampling少数类
4. **迁移学习**：从相似任务fine-tune

### 提升准确率的方法

1. **模型集成**：训练多个模型投票
2. **Test-Time Augmentation (TTA)**：测试时增强
3. **更大的模型**：yolo11l-cls或yolo11x-cls
4. **更多训练数据**：收集更多logo图片
5. **调整超参数**：网格搜索最优参数

## 📚 参考资料

- [Ultralytics官方文档](https://docs.ultralytics.com/)
- [YOLOv11分类教程](https://docs.ultralytics.com/tasks/classify/)
- [模型训练配置](https://docs.ultralytics.com/modes/train/)
- [数据增强技巧](https://docs.ultralytics.com/usage/cfg/)

---

**准备好了吗？运行 `check_environment.py` 开始吧！** 🚀
