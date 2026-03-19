# YOLOv11 Logo Classification Complete Workflow

## Provided Files

1. **prepare_yolo_dataset_enhanced.py** - Dataset preparation script (already run)
2. **train_yolo_classifier.py** - Full training script
3. **train_simple.py** - Simplified training script
4. **train_logo_oneclick.sh** - One-click training script (recommended)
5. **predict_logo.py** - Inference prediction script
6. **check_environment.py** - Environment check script
7. **README_YOLO_TRAINING.md** - Detailed usage documentation

## Quick Start (3 Steps)

### Step 1: Check Environment

```bash
cd /home1/lu-wei/repo/EMMA/classifier/YOLO
python check_environment.py
```

If it shows "All checks passed", continue to the next step.

### Step 2: Start Training

```bash
# Option 1: One-click training (simplest)
./train_logo_oneclick.sh

# Option 2: Python script
python train_simple.py

# Option 3: Command line
yolo classify train \
    data=logo_dataset_35_enhanced \
    model=yolo11s-cls.pt \
    epochs=100 \
    imgsz=224 \
    batch=32 \
    device=0
```

### Step 3: Validate and Predict

```bash
# Validate model
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --mode val

# Predict a single image
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source test.jpg \
    --mode single
```

## Training Configuration Suggestions

### Choose Configuration Based on GPU Memory

| GPU Memory | Model | Batch Size | Estimated Time |
|------|------|------------|----------|
| 4GB | yolo11n-cls | 8-16 | 60-90 minutes |
| 8GB | yolo11s-cls | 16-32 | 45-60 minutes |
| 12GB | yolo11m-cls | 32-64 | 60-90 minutes |
| 24GB | yolo11l-cls | 64-128 | 90-120 minutes |

**Recommended configuration**:
- Model: **yolo11s-cls** or **yolo11m-cls**
- Batch size: **32**
- Epochs: **100**
- Expected Top-1 accuracy: **75-85%**

## Directory Structure

```
/home1/lu-wei/repo/EMMA/classifier/YOLO/
│
├── logo_dataset_35_enhanced/       ← Dataset (already prepared)
│   ├── train/                      ← Training set: 2849 images
│   │   ├── Apple/
│   │   ├── BMW/
│   │   └── ... (35 brands)
│   └── test/                       ← Test set: 729 images
│       ├── Apple/
│       └── ...
│
├── src/                            ← Code files
│   ├── prepare_yolo_dataset_enhanced.py
│   ├── train_yolo_classifier.py
│   ├── train_simple.py
│   ├── train_logo_oneclick.sh
│   ├── predict_logo.py
│   └── check_environment.py
│
└── runs/classify/                  ← Training output (generated after training)
    └── logo_35_xxx/
        ├── weights/
        │   ├── best.pt            ← Best model
        │   └── last.pt            ← Last model
        ├── results.png            ← Training curves
        ├── confusion_matrix.png   ← Confusion matrix
        └── results.csv            ← Detailed results
```

## Key Parameter Descriptions

```python
# Basic parameters
model = 'yolo11s-cls.pt'    # Model size
epochs = 100                 # Number of training epochs
imgsz = 224                  # Image size (standard)
batch = 32                   # Batch size
device = 0                   # GPU number

# Learning rate
lr0 = 0.01                   # Initial learning rate
lrf = 0.01                   # Final learning rate factor

# Early stopping
patience = 20                # Stop if no improvement for 20 epochs

# Data augmentation
fliplr = 0.5                 # Horizontal flip
hsv_h = 0.015               # Hue
hsv_s = 0.7                 # Saturation
hsv_v = 0.4                 # Value (brightness)
degrees = 10                 # Rotation angle
scale = 0.5                  # Scaling
```

## Monitoring Training

Real-time display during training:

```
Epoch    GPU_mem   loss  top1_acc  top5_acc
  1/100     2.5G   3.52     0.15      0.45
  2/100     2.5G   2.98     0.28      0.62
  3/100     2.5G   2.45     0.42      0.75
  ...
100/100     2.5G   0.85     0.82      0.96
```

Key metrics:
- **loss**: Lower is better
- **top1_acc**: Top-1 accuracy (main metric)
- **top5_acc**: Top-5 accuracy

## Evaluating Results

After training, view results:

```bash
cd runs/classify/logo_35_xxx

# View final metrics
cat results.csv | tail -5

# View confusion matrix
display confusion_matrix.png

# View training curves
display results.png

# View model size
ls -lh weights/
```

## FAQ

### 1. CUDA out of memory
```bash
# Reduce batch size
batch=16  # or 8
```

### 2. Training is slow
```bash
# Confirm GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Use a smaller model
model=yolo11n-cls.pt
```

### 3. Low accuracy
- Increase training epochs: epochs=150
- Use a larger model: yolo11m-cls.pt
- Check data quality and labels

### 4. Overfitting
- Increase data augmentation
- Use a smaller model
- Early stopping: patience=20

## Post-Training Operations

### Export Model
```python
from ultralytics import YOLO
model = YOLO('runs/classify/logo_35_xxx/weights/best.pt')
model.export(format='onnx')  # Export to ONNX
```

### Test on New Images
```bash
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source new_logo.jpg \
    --mode single
```

### Batch Inference
```bash
python predict_logo.py \
    --model runs/classify/logo_35_xxx/weights/best.pt \
    --source test_images/ \
    --mode batch
```

## Optimization Suggestions

### If Data is Imbalanced
Some classes have too few images (e.g., Logitech with only 7):

1. **Data augmentation**: Increase augment parameters
2. **Class weights**: Use weighted loss
3. **Resampling**: Oversample minority classes
4. **Transfer learning**: Fine-tune from a similar task

### Methods to Improve Accuracy

1. **Model ensemble**: Train multiple models and vote
2. **Test-Time Augmentation (TTA)**: Augment during testing
3. **Larger model**: yolo11l-cls or yolo11x-cls
4. **More training data**: Collect more logo images
5. **Hyperparameter tuning**: Grid search for optimal parameters

## References

- [Ultralytics Official Documentation](https://docs.ultralytics.com/)
- [YOLOv11 Classification Tutorial](https://docs.ultralytics.com/tasks/classify/)
- [Model Training Configuration](https://docs.ultralytics.com/modes/train/)
- [Data Augmentation Tips](https://docs.ultralytics.com/usage/cfg/)

---

**Ready? Run `check_environment.py` to get started!**
