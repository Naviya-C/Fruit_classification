# 🍎 Fruit Classification using Transfer Learning (Model Comparison)

## 📌 Overview

This project focuses on identifying the **best-performing deep learning model** among multiple architectures using **transfer learning without fine-tuning**.

The following pretrained CNN models are compared:

* ResNet18
* ResNet50
* EfficientNet-B0

👉 The objective is to evaluate how effectively pretrained feature extractors perform when **only the final classification layer is trained**, without updating backbone weights.

---

## 🎯 Objective

* Apply **transfer learning** using pretrained models (trained on ImageNet)
* Freeze backbone layers and train only the classifier
* Compare multiple architectures under the same conditions
* Evaluate models using robust performance metrics
* Select the best model based on validation/test results

---

## 🧠 Models Used

### 1. ResNet18

* Lightweight and fast to train
* Good baseline model
* Suitable for smaller datasets

### 2. ResNet50

* Deeper architecture with higher representational capacity
* Uses residual connections to avoid vanishing gradients
* Extracts richer features than ResNet18

### 3. EfficientNet-B0

* Uses compound scaling (depth, width, resolution)
* Optimized for efficiency vs accuracy
* Strong performance with fewer parameters

---

## ⚙️ Methodology

### 🔹 Transfer Learning (No Fine-Tuning)

For all models:

* Loaded pretrained weights from ImageNet
* Froze all backbone layers:

```python
for param in model.parameters():
    param.requires_grad = False
```

* Replaced final classification layer:

  * ResNet → `model.fc`
  * EfficientNet → `model.classifier[1]`

* Trained **only the final layer**

👉 This ensures:

* Faster convergence
* Reduced overfitting
* Fair and controlled comparison across models

---

## 🧪 Dataset

* Fruits-360 dataset (original size)
* Organized into:

  * Training set
  * Validation set
  * Test set

### Data Augmentation (Training Only)

* Random Horizontal Flip
* Random Rotation
* Color Jitter

### Normalization

* ImageNet mean and standard deviation

---

## 📊 Evaluation Metrics

Each model is evaluated using:

* **Accuracy** — overall correctness
* **Precision** — correctness of positive predictions
* **Recall** — ability to capture all true instances
* **F1 Score** — balance between precision and recall

👉 Using multiple metrics provides a more reliable evaluation for multi-class classification.

---

## 📈 Results

| Model           | Accuracy   | Precision  | Recall     | F1 Score   |
| --------------- | ---------- | ---------- | ---------- | ---------- |
| ResNet18        | 0.9219     | 0.9237     | 0.9219     | 0.9222     |
| ResNet50        | **0.9294** | **0.9293** | **0.9294** | **0.9290** |
| EfficientNet-B0 | 0.9121     | 0.9141     | 0.9121     | 0.9116     |

👉 **Best Model: ResNet50** (highest overall performance)

---

## 🧠 Why ResNet & EfficientNet?

### ✔ ResNet

* Introduced residual (skip) connections to stabilize deep training
* Proven, reliable architecture for image classification
* Strong baseline across many tasks

### ✔ EfficientNet

* Uses compound scaling for better efficiency
* Achieves strong performance with fewer parameters
* Designed to balance accuracy and computational cost

👉 Comparing both provides insight into:

* Traditional deep CNN architectures vs modern optimized architectures

---

## 🏗️ Project Structure

```
fruit-classification-ml/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── model_architecture.py
│
├── preprocess/
│   └── data_load.py
│
├── models/
├── results/
│   ├── accuracy.txt
│   └── evaluation.txt
│
├── app/
│   └── main.py
│
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
python src/train.py
```

### 3. Evaluate model

```bash
python src/evaluate.py
```

---

## ⚡ Special Implementation Details

* ✔ Model-agnostic training pipeline (supports multiple architectures)
* ✔ Only trainable parameters passed to optimizer:

```python
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0003
)
```

* ✔ Best model saved based on validation accuracy
* ✔ Multi-GPU support using `DataParallel`

---

## 🧠 Key Takeaways

* Transfer learning is effective even without fine-tuning
* ResNet50 achieved the best performance in this setup
* EfficientNet is efficient but not always superior without fine-tuning
* Validation-based model selection is critical
* Multiple evaluation metrics provide a more complete performance view

---

## 🚀 Future Improvements

* Fine-tune deeper layers for improved accuracy
* Add confusion matrix visualization
* Implement training/validation curves
* Deploy using Streamlit
* Add top-k prediction confidence

---

## 👨‍💻 Author

This project was developed to gain hands-on experience in transfer learning and model comparison using real-world datasets.

---
