# YOLOv8 Logo分类 - 数据准备指南

## 📋 任务目标

从LogoDet-3K数据集中提取35个目标品牌的图片，并按照YOLO格式组织数据集，用于训练YOLOv8分类模型。

## 🎯 35个目标品牌

```python
target_brands = {
    'Food': ['Heineken', 'nestle', 'GUINNESS', 'Coca-Cola', "McDonald's"],          # 5个
    'Clothes': ['Asics', 'Gap', 'Converse', 'Calvin Klein', 'Hermes'],             # 5个
    'Necessities': ['Colgate', 'nivea', 'Gillette', 'pantene', 'neutrogena'],      # 5个
    'Electronic': ['Apple', 'Canon', 'Logitech', 'ASUS', 'HTC'],                   # 5个
    'Transportation': ['BMW', 'lexus', 'Lamborghini', 'Chevrolet', 'michelin'],    # 5个
    'Leisure': ['Marvel', 'Barbie', 'Hot Wheels', 'Mattel', 'play-doh'],          # 5个
    'Sports': ['spalding', 'oakley', 'under armour', 'wilson', 'yonex']            # 5个
}
```

**总计：35个品牌**

## 📁 数据集结构

### 原始数据位置：
```
/home1/lu-wei/repo/EMMA/classifier/YOLO/VOC3000number/
├── JPEGImages/          # 158,654张图片
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── Annotations/         # XML标注文件
│   ├── 000001.xml
│   ├── 000002.xml
│   └── ...
└── logo_num.txt        # 品牌名称和数量映射
```

### 目标数据结构：
```
logo_dataset_35/
├── train/              # 训练集 (80%)
│   ├── Heineken/
│   │   ├── 000123.jpg
│   │   ├── 000456.jpg
│   │   └── ...
│   ├── Apple/
│   │   ├── 001234.jpg
│   │   └── ...
│   ├── BMW/
│   └── ...
│
└── test/               # 测试集 (20%)
    ├── Heineken/
    ├── Apple/
    ├── BMW/
    └── ...
```

## 🚀 使用步骤

### 步骤1: 检查品牌是否在数据集中

```bash
cd /home1/lu-wei/repo/EMMA/classifier/YOLO/
python check_brands.py
```

这个脚本会：
- 检查35个品牌是否在LogoDet-3K数据集中
- 显示每个品牌的图片数量
- 列出未找到的品牌

### 步骤2: 准备数据集

```bash
python prepare_yolo_dataset.py
```

这个脚本会：
1. 读取logo_num.txt文件，获取品牌ID映射
2. 在数据集中查找35个目标品牌
3. 解析所有XML文件，提取目标品牌的图片
4. 按80/20比例划分训练集和测试集
5. 将图片复制到对应的文件夹

### 步骤3: 验证数据集

```bash
# 查看数据集结构
tree -L 2 logo_dataset_35/

# 统计各品牌图片数量
for brand in logo_dataset_35/train/*/; do 
    echo "$brand: $(ls -1 $brand | wc -l) images"
done
```

## ⚙️ 配置参数

在 `prepare_yolo_dataset.py` 中可以修改以下参数：

```python
# 数据路径
base_dir = "/home1/lu-wei/repo/EMMA/classifier/YOLO/VOC3000number"
output_dir = "/home1/lu-wei/repo/EMMA/classifier/YOLO/logo_dataset_35"

# 训练集比例 (默认80%)
train_ratio = 0.8

# 随机种子 (保证可复现)
random.seed(42)
```

## 📊 预期输出

脚本执行完成后会显示：

```
================================================================================
数据集准备完成 - 统计报告
================================================================================

1. 品牌匹配情况:
   - 目标品牌数: 35
   - 找到的品牌: XX
   - 未找到的品牌: XX

2. 图片统计:
   - 总图片数: XX,XXX
   - 训练集: XX,XXX
   - 测试集: X,XXX

3. 各类别统计:
   Food           : XX,XXX 张
   Clothes        : XX,XXX 张
   ...

4. 品牌数量排序 (Top 10):
   Brand1         : X,XXX 张
   Brand2         : X,XXX 张
   ...
```

## 🔧 训练YOLOv8分类模型

数据准备完成后，可以使用以下代码训练模型：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n-cls.pt')  # nano模型，可选 s, m, l, x

# 训练模型
results = model.train(
    data='logo_dataset_35',      # 数据集路径
    epochs=100,                  # 训练轮数
    imgsz=224,                   # 图片大小
    batch=32,                    # 批次大小
    device=0,                    # GPU设备
    project='logo_classifier',   # 项目名称
    name='yolo_logo_35'         # 实验名称
)

# 验证模型
metrics = model.val()

# 使用模型
results = model('test_image.jpg')
print(results[0].probs.top1)  # 预测类别
```

## ⚠️ 注意事项

1. **品牌名称匹配**：
   - 某些品牌可能在数据集中使用不同的名称
   - 脚本会进行模糊匹配，但可能需要手动调整

2. **数据平衡**：
   - 不同品牌的图片数量可能差异很大
   - 训练时可能需要使用类别权重或数据增强

3. **存储空间**：
   - 复制图片需要足够的磁盘空间
   - 可以选择使用符号链接代替复制（修改脚本中的shutil.copy2）

4. **处理时间**：
   - 处理158,654个XML文件需要一些时间
   - 复制图片也需要时间

## 📝 可能遇到的问题

### 问题1: 某些品牌未找到

**解决方案**：
- 检查品牌名称是否正确
- 在logo_num.txt中搜索相似名称
- 可以手动添加映射关系

### 问题2: 图片路径错误

**解决方案**：
- 确认JPEGImages文件夹名称正确
- 检查文件名后缀（.jpg vs .JPG）

### 问题3: XML解析错误

**解决方案**：
- 脚本会自动跳过损坏的XML文件
- 检查Annotations文件夹权限

## 📚 参考资源

- LogoDet-3K论文: https://arxiv.org/abs/2008.05359
- YOLOv8文档: https://docs.ultralytics.com
- YOLO分类教程: https://docs.ultralytics.com/tasks/classify/

## 📧 联系方式

如有问题，请查看脚本输出的详细日志。
