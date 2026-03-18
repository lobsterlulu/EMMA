# 📦 YOLOv11 Logo分类 - 完整文件包

## 文件清单

### 📘 文档文件
1. **QUICKSTART.md** - 快速开始指南（⭐ 从这里开始）
2. **README_YOLO_TRAINING.md** - 详细训练文档

### 🔧 准备阶段
3. **prepare_yolo_dataset_enhanced.py** - 数据集准备脚本（已运行完成）
4. **check_environment.py** - 环境检查脚本

### 🚀 训练阶段
5. **train_logo_oneclick.sh** - 一键训练脚本（⭐ 推荐）
6. **train_simple.py** - 简化Python训练脚本
7. **train_yolo_classifier.py** - 完整Python训练脚本

### 🎯 推理阶段
8. **predict_logo.py** - 推理预测脚本

---

## 使用流程

### ✅ 第1步：复制文件到服务器

```bash
# 在服务器上创建目录
mkdir -p /home1/lu-wei/repo/EMMA/classifier/YOLO/src

# 将下载的文件复制到此目录
# QUICKSTART.md
# README_YOLO_TRAINING.md
# check_environment.py
# train_logo_oneclick.sh
# train_simple.py
# train_yolo_classifier.py
# predict_logo.py
```

### ✅ 第2步：设置执行权限

```bash
cd /home1/lu-wei/repo/EMMA/classifier/YOLO/src
chmod +x train_logo_oneclick.sh
```

### ✅ 第3步：检查环境

```bash
python check_environment.py
```

预期输出：
```
======================================================================
✓ 所有检查通过！可以开始训练了
======================================================================
```

### ✅ 第4步：开始训练

```bash
# 最简单的方式
./train_logo_oneclick.sh
```

### ✅ 第5步：训练完成后

查看结果：
```bash
cd runs/classify/logo_35_yolo11s
ls -lh weights/
cat results.csv
```

验证模型：
```bash
python predict_logo.py \
    --model runs/classify/logo_35_yolo11s/weights/best.pt \
    --mode val
```

---

## 文件详细说明

### 1. QUICKSTART.md
- 3步快速开始
- 配置建议
- 目录结构
- 常见问题

### 2. README_YOLO_TRAINING.md
- 完整训练指南
- 参数详解
- 性能优化
- 导出部署

### 3. prepare_yolo_dataset_enhanced.py
✅ **已运行完成**
- 从LogoDet-3K数据集提取35个品牌
- 创建train/test目录结构
- 输出：logo_dataset_35_enhanced/

### 4. check_environment.py
在训练前运行，检查：
- Python包安装
- CUDA/GPU可用性
- 数据集完整性
- 模型下载

### 5. train_logo_oneclick.sh ⭐
**推荐使用**
- Bash脚本，一键启动训练
- 自动检查环境
- 显示训练进度
- 训练完成后显示结果路径

使用方法：
```bash
./train_logo_oneclick.sh
```

### 6. train_simple.py
Python简化版训练脚本
- 最少的配置
- 适合快速测试
- 7行代码开始训练

使用方法：
```bash
python train_simple.py
```

### 7. train_yolo_classifier.py
Python完整版训练脚本
- 详细的配置选项
- 自动检查数据集
- 训练+验证+测试
- 包含注释和说明

使用方法：
```bash
python train_yolo_classifier.py
```

### 8. predict_logo.py
推理预测脚本
- 单张图片预测
- 批量图片预测
- 测试集验证
- 显示Top-K结果

使用方法：
```bash
# 单张图片
python predict_logo.py \
    --model weights/best.pt \
    --source test.jpg \
    --mode single

# 批量预测
python predict_logo.py \
    --model weights/best.pt \
    --source test_images/ \
    --mode batch

# 验证测试集
python predict_logo.py \
    --model weights/best.pt \
    --mode val
```

---

## 数据集信息

### 已准备的数据集
- **路径**: `/home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35_enhanced`
- **类别数**: 35个品牌
- **训练集**: ~2849张图片
- **测试集**: ~729张图片

### 类别列表
Food (5): Heineken, nestle, GUINNESS, Coca-Cola, McDonald's
Clothes (5): Asics, Gap, Converse, Calvin Klein, Hermes
Necessities (5): Colgate, nivea, Gillette, pantene, neutrogena
Electronic (5): Apple, Canon, Logitech, ASUS, HTC
Transportation (5): BMW, lexus, Lamborghini, Chevrolet, michelin
Leisure (5): Marvel, Barbie, Hot Wheels, Mattel, play-doh
Sports (5): spalding, oakley, under armour, wilson, yonex

---

## 训练配置推荐

### 标准配置（推荐）
```
模型: yolo11s-cls.pt
Epochs: 100
Batch: 32
Image Size: 224
Learning Rate: 0.01
Patience: 20
预计时间: 45-60分钟
预计准确率: 75-85%
```

### 快速测试
```
模型: yolo11n-cls.pt
Epochs: 50
Batch: 16
预计时间: 20-30分钟
预计准确率: 70-80%
```

### 高精度
```
模型: yolo11m-cls.pt
Epochs: 150
Batch: 32
预计时间: 90-120分钟
预计准确率: 80-88%
```

---

## 预期结果

### 训练指标
- **Loss**: 0.5-1.0
- **Top-1 Accuracy**: 75-85%
- **Top-5 Accuracy**: 90-95%

### 输出文件
训练完成后在 `runs/classify/logo_35_xxx/` 目录：
- `weights/best.pt` - 最佳模型
- `weights/last.pt` - 最后一次训练
- `results.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵
- `results.csv` - 详细数据

---

## 下一步

1. ✅ 复制所有文件到服务器
2. ✅ 运行 `check_environment.py`
3. ✅ 运行 `train_logo_oneclick.sh`
4. ✅ 等待训练完成（~60分钟）
5. ✅ 验证模型准确率
6. ✅ 使用 `predict_logo.py` 测试

---

## 支持和帮助

遇到问题？
1. 先查看 **QUICKSTART.md** 的"常见问题"部分
2. 查看 **README_YOLO_TRAINING.md** 获取详细说明
3. 检查训练日志：`runs/classify/logo_35_xxx/`

---

祝训练顺利！🚀
