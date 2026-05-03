import torch
from torchvision import transforms
from PIL import Image
from model_architecture import get_resnet_model

MODEL_PATH = "models/resnet_model.pth"
CLASSES = ["apple", "banana", "orange", "mango", "grape"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_resnet_model(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    return CLASSES[predicted.item()]