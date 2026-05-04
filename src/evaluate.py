# src/evaluate.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from model_architecture import ResNet, EfficientNet
from data_load import test_load, num_class
import os

# -----------------------
# Config
# -----------------------
MODEL_NAME = "resnet50"  # change if needed
MODEL_PATH = f"/home/naviya-c/Desktop/Projects/fruit-classification-ml/models/{MODEL_NAME}_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Model
# -----------------------
num_classes = num_class()

if MODEL_NAME == "resnet18":
    model = ResNet.get_resnet_model18(num_classes)

elif MODEL_NAME == "resnet50":
    model = ResNet.get_resnet_model50(num_classes)

elif MODEL_NAME == "efficientnet_b0":
    model = EfficientNet.get_efficientnet_b0(num_classes)

else:
    raise ValueError("Invalid model")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -----------------------
# Evaluation
# -----------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_load:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -----------------------
# Metrics
# -----------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n📊 Evaluation Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds))

# -----------------------
# Save Results
# -----------------------
os.makedirs("results", exist_ok=True)

with open("results/evaluation.txt", "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")

print("\n✅ Evaluation saved to results/evaluation.txt")