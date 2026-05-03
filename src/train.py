import torch
import torch.nn as nn
import torch.optim as optim
import os

from data_load import train_load, val_load, num_class
from model_architecture import ResNet, EfficientNet

# -----------------------
# Config
# -----------------------
MODEL_NAME = "efficientnet_b0"   # "resnet18", "resnet50", "efficientnet_b0"
EPOCHS = 10
LR = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Model Selection
# -----------------------
num_classes = num_class()

if MODEL_NAME == "resnet18":
    model = ResNet.get_resnet_model18(num_classes)

elif MODEL_NAME == "resnet50":
    model = ResNet.get_resnet_model50(num_classes)

elif MODEL_NAME == "efficientnet_b0":
    model = EfficientNet.get_efficientnet_b0(num_classes)

else:
    raise ValueError("Invalid model name")

# -----------------------
# Multi-GPU
# -----------------------
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# -----------------------
# Loss & Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()

# only train unfrozen layers
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# -----------------------
# Training Loop
# -----------------------
best_val_acc = 0

for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_load:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    avg_loss = train_loss / len(train_load)

    # ===== VALIDATION =====
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_load:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    # ===== SAVE BEST MODEL =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        os.makedirs("models", exist_ok=True)
        save_path = f"models/{MODEL_NAME}_best.pth"

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

    # ===== LOG =====
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

# -----------------------
# Save Final Model
# -----------------------
final_path = f"models/{MODEL_NAME}_final.pth"

if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), final_path)
else:
    torch.save(model.state_dict(), final_path)

# -----------------------
# Save Results
# -----------------------
os.makedirs("results", exist_ok=True)

with open("results/accuracy.txt", "w") as f:
    f.write(f"{MODEL_NAME} Best Validation Accuracy: {best_val_acc*100:.2f}%\n")

print("\nTraining complete. Models saved.")