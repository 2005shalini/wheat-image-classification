# Wheat Seed Variety Classification

Deep learning image classification model to identify wheat seed varieties (Akbar, Dilkash, Urooj) using transfer learning with MobileNetV2.

## Project Overview

This project classifies wheat seeds into three varieties based on RGB images using a Convolutional Neural Network built on top of a pretrained MobileNetV2 backbone (transfer learning + fine-tuning).

## Dataset

- **Source:** [Varietal Purity of Wheat Seeds Dataset](https://data.mendeley.com/datasets/w5248v9fk3/1), Mendeley Data, University of Agriculture, Faisalabad
- **Classes:** Akbar, Dilkash, Urooj (3 wheat varieties)
- **Samples:** 375 unique seeds (125 per class), 1124 total images (each seed has 1-4 photos at different angles)
- **Image specs:** 256×256 pixels, RGB

> Note: Raw dataset images are not included in this repository due to size. Download from the link above and place inside `Dataset/Wheat varieties dataset/` to reproduce results. A pre-built index (`Dataset/dataset_index_with_splits.csv`) is included for reference.

## Project Structure

Wheat-Image-Classification/
├── Dataset/ # Dataset index files
├── models/ # Trained model (best_model.keras)
├── notebooks/
│ ├── 01_data_inspection.ipynb # Dataset verification, EDA, train/val/test split
│ └── 02_model_building.ipynb # Model building, training, evaluation
├── src/
│ └── inference.py # Standalone inference script
├── results/ # Confusion matrix, training curves, sample predictions
└── README.md


## Methodology

1. **Data Verification:** Programmatic inspection of image counts, dimensions, color modes, corrupted files
2. **Preprocessing:** Images resized to 256×256, normalized via MobileNetV2 preprocessing
3. **Augmentation:** Random flips, rotation (15%), brightness/contrast variation (training set only)
4. **Splitting:** Seed-level stratified split (70/15/15 train/val/test) to prevent data leakage between multiple images of the same seed
5. **Model:** MobileNetV2 (ImageNet pretrained) + custom classification head, initially trained with frozen base, then fine-tuned by unfreezing top 54 layers

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 79.65% |
| Precision (macro) | 80.41% |
| Recall (macro) | 79.63% |
| F1-score (macro) | 79.85% |

- Best performance on **Akbar** class (94% precision) — visually distinct variety
- **Dilkash** and **Urooj** show higher mutual confusion, suggesting closer visual similarity

See `results/confusion_matrix.png` and `results/training_curves.png` for details.

## How to Run Inference

```bash
cd src
python3 inference.py "path/to/your/wheat_seed_image.jpg"
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow jupyter
```

## Limitations & Future Scope

- Small dataset size (375 seeds) limits generalization; larger datasets would likely improve accuracy
- Dilkash/Urooj confusion suggests need for additional distinguishing features or higher-resolution imagery
- Future work: expand to more wheat varieties, test additional architectures (EfficientNet, ResNet), deploy as a web/mobile app