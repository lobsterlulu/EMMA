# Classifier

Inference-only classifiers used to evaluate concept erasure across five domains. Each subdirectory contains only the core classification pipeline — no training scripts, attack code, or evaluation harness.

| Directory | Domain | Classifier |
|-----------|--------|------------|
| [`ML_Decoder/`](#ml_decoder) | Object, NSFW | ML-Decoder |
| [`GCD/`](#gcd) | Celebrity | Giphy Celebrity Detector |
| [`Diffusion-MU-Attack/`](#diffusion-mu-attack) | Art Style | UnlearnDiffAtk |
| [`NudeNet/`](#nudenet) | NSFW | NudeNet |
| [`Q16/`](#q16) | NSFW | Q16 |
| [`YOLO/`](#yolo) | Copyright | YOLOv11 |

---

## ML_Decoder

Multi-label classification head based on cross-attention between learned queries and image features. Used for COCO object detection.

- Paper: [ML-Decoder: Scalable and Versatile Classification Head](https://arxiv.org/abs/2111.12933)
- Code: [Alibaba-MIIL/ML_Decoder](https://github.com/Alibaba-MIIL/ML_Decoder)

**Contents:**
```
ML_Decoder/
├── src_files/ml_decoder/ml_decoder.py   # Core attention head architecture
├── inplace_abn/                          # In-place batch norm dependency
├── infer.py                              # Inference: COCO object detection
├── infer_nsfw.py                         # Inference: COCO object classification in NSFW domain
├── validate.py                           # Validation/evaluation
└── detected_classes.txt                  # 80 detectable class labels
```

---

## GCD

Face detection + recognition pipeline from Giphy's open-source celebrity detector. Detects faces via MTCNN, then classifies identity using a fine-tuned ResNet model.

- Code: [Giphy/celeb-detection-oss](https://github.com/Giphy/celeb-detection-oss)

**Contents:**
```
GCD/
├── evaluate_by_GCD.py       # Entry point: classify celebrity identity in an image folder
├── model_training/          # Face detector + recognizer library
└── resources/
    ├── face_detection/      # MTCNN weights (det1/2/3.npy)
    └── face_recognition/    # Trained model (best_model_states.pkl) + labels.csv
```

> **Note:** `evaluate_by_GCD.py` sets `APP_DATA_DIR` to an absolute path. Update it (or the `.env` file) to point to `GCD/resources/` before running.

---

## Diffusion-MU-Attack

Art style classifier fine-tuned on 129 artist styles (Monet, Warhol, Cézanne, Rembrandt, etc.), built on a HuggingFace `image-classification` pipeline over a fine-tuned ResNet-50.

- Paper: [To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now](https://arxiv.org/abs/2310.11868)
- Code: [OPTML-Group/Diffusion-MU-Attack](https://github.com/OPTML-Group/Diffusion-MU-Attack)

**Contents:**
```
Diffusion-MU-Attack/
├── src/
│   ├── art_classifier.py          # Entry point: batch art style classification with per-artist accuracy
│   ├── img_batch_classify.py      # Single-folder style/object/nudity classifier
│   └── utils/metrics/
│       └── style_eval.py          # Core: HuggingFace pipeline wrapper
└── classifier/
    └── checkpoint-2800/           # Fine-tuned ResNet-50 weights (129 art styles)
```

---

## NudeNet

**Domain:** NSFW

Nudity detector using [NudeNet](https://github.com/notAI-tech/NudeNet)'s pre-trained `NudeDetector`. Flags exposed body parts from a predefined set of sensitive classes and outputs per-image results to CSV.

- Code: [notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet)

**Contents:**
```
NudeNet/
└── evaluate_by_nudenet.py   # Entry point: detect nudity in an image folder, save CSV
```

**Dependency:** `pip install nudenet` (downloads model weights automatically).

---

## Q16

CLIP-based binary classifier for inappropriate content (Q16 probe). Uses pre-computed text embeddings from CLIP to classify images as appropriate or inappropriate via cosine similarity.

- Paper: [Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models](https://arxiv.org/abs/2305.13873)
- Code: [YitingQu/unsafe-diffusion](https://github.com/YitingQu/unsafe-diffusion)

**Contents:**
```
Q16/
├── classify/
│   ├── inference_images.py        # Entry point: classify images in a folder
│   ├── inference_embeddings.py    # Entry point: classify via pre-computed embeddings
│   └── utils.py                   # ClipWrapper, SimClassifier, load_prompts
├── models/
│   ├── clip.py                    # ClipVisionModel, ClipSimModel, ClipSimModel_Infer
│   └── baseline.py                # ResNet50 baseline alternative
└── data/
    ├── ViT-B-16/prompts.p
    ├── ViT-B-32/prompts.p
    └── ViT-L-14/prompts.p         # Pre-trained text embeddings
```

---

## YOLO

YOLOv11-based logo classifier for brand/copyright evaluation. Custom-trained on 30 brand logo classes to detect whether generated images contain recognizable brand logos.

- Framework: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

**Training details:**

The model was trained from scratch (fine-tuned from `yolo11*-cls.pt` pretrained weights) on a curated logo dataset. Training data, scripts, and run history are preserved in the original working directory (`classifier_ori/YOLO/`).

- **Dataset:** ~4,600 images across 30 brand classes (e.g., Apple, BMW, Coca-Cola, McDonald's, Lamborghini, Marvel, etc.), split into train (~2,870) and test (~1,711) sets.
- **Architecture:** YOLOv11 classification head (multiple sizes available: `yolo11n-cls` through `yolo11x-cls`).
- **Image size:** 224 × 224
- **Key training params:** 100 epochs, batch 32, patience 20 (early stopping), lr0=0.01

**Contents:**
```
YOLO/
├── run.py                          # Entry point: classify images in a folder, output CSV
├── src/
│   ├── predict_logo.py             # Single-image logo prediction
│   ├── batch_predict_logos.py      # Batch logo prediction
│   └── batch_predict_metrics.py    # Batch evaluation with metrics
├── model/                          # Trained YOLOv11 model weights (.pt)
└── QUICKSTART.md                   # Training & inference guide
```
