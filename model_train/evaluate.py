from sklearn.metrics import accuracy_score, f1_score
from model_train.model_setup import transfer_model
from preprocess.data_load import test_load 
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of classes
num_classes = len(test_load.dataset.classes)

# Load trained model
model = transfer_model(num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_load:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds, average='weighted'))