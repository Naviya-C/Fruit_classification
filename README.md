<<<<<<< HEAD
# 🍎 Fruit Classification using ResNet50 (Transfer Learning)

## 📌 Project Overview
This project focuses on classifying fruits using a deep learning model based on **ResNet50** with transfer learning. The model is trained on a labeled fruit image dataset to accurately predict the category of a given fruit image.

The objective is to build a robust image classification system that can generalize well to unseen data by leveraging pretrained weights.

---

## 🚀 Features
- Fruit classification using ResNet50
- Transfer learning with pretrained ImageNet weights
- Fine-tuning for improved accuracy
- Supports image-based inference
- Modular and scalable training pipeline

---

## 🧠 Model Details
- **Model:** ResNet50
- **Task:** Image Classification
- **Training Type:** Transfer Learning + Fine-tuning
- **Framework:** PyTorch

---

## 📂 Dataset
- Custom fruit dataset with labeled images
- Organized into class-wise folders
- Example structure:
```bash
dataset/
  ├── train/
  │   ├── apple 2/
  │   ├── apple 1/
  │   ├── orange/
  ├── val/
      ├── apple 2/
      ├── apple 1/
      ├── orange/
=======
🍎 Fruit Classification using Transfer Learning (Model Comparison)
📌 Overview

This project focuses on identifying the best-performing deep learning model among multiple architectures using transfer learning without fine-tuning.

The following pretrained CNN models are compared:

ResNet18
ResNet50
EfficientNet-B0

👉 The objective is to evaluate how effectively pretrained feature extractors perform when only the final classification layer is trained, without updating backbone weights.

🎯 Objective
Apply transfer learning using pretrained models (trained on ImageNet)
Freeze backbone layers and train only the classifier
Compare multiple architectures under the same conditions
Evaluate models using robust performance metrics
Select the best model based on validation/test results
🧠 Models Used
1. ResNet18
Lightweight and fast to train
Good baseline model
Suitable for smaller datasets
2. ResNet50
Deeper architecture with higher representational capacity
Uses residual connections to avoid vanishing gradients
Extracts richer features than ResNet18
3. EfficientNet-B0
Uses compound scaling (depth, width, resolution)
Optimized for efficiency vs accuracy
Strong performance with fewer parameters
⚙️ Methodology
🔹 Transfer Learning (No Fine-Tuning)

For all models:

Loaded pretrained weights from ImageNet
Froze all backbone layers:
for param in model.parameters():
    param.requires_grad = False
Replaced final classification layer:
ResNet → model.fc
EfficientNet → model.classifier[1]
Trained only the final layer

👉 This ensures:

Faster convergence
Reduced overfitting
Fair and controlled comparison across models
🧪 Dataset
Fruits-360 dataset (original size)
Organized into:
Training set
Validation set
Test set
Data Augmentation (Training Only)
Random Horizontal Flip
Random Rotation
Color Jitter
Normalization
ImageNet mean and standard deviation
📊 Evaluation Metrics

Each model is evaluated using:

Accuracy — overall correctness
Precision — correctness of positive predictions
Recall — ability to capture all true instances
F1 Score — balance between precision and recall

👉 Using multiple metrics provides a more reliable evaluation for multi-class classification.

📈 Results
Model	Accuracy	Precision	Recall	F1 Score
ResNet18	0.9219	0.9237	0.9219	0.9222
ResNet50	0.9294	0.9293	0.9294	0.9290
EfficientNet-B0	0.9121	0.9141	0.9121	0.9116

👉 Best Model: ResNet50 (highest overall performance)

🧠 Why ResNet & EfficientNet?
✔ ResNet
Introduced residual (skip) connections to stabilize deep training
Proven, reliable architecture for image classification
Strong baseline across many tasks
✔ EfficientNet
Uses compound scaling for better efficiency
Achieves strong performance with fewer parameters
Designed to balance accuracy and computational cost

👉 Comparing both provides insight into:

Traditional deep CNN architectures vs modern optimized architectures
>>>>>>> 143e560 (Re-structure and make readme.md file)
