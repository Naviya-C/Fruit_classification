from torchvision import transforms
import torch
from torchvision.models import resnet50
from PIL import Image
import io


check_point = torch.load(
    "app/backend/resnet50_final.pth",
    map_location = "cpu"
)

classes = check_point["model_names"]
num_classes = len(classes)

model = resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(
    check_point["model_state"]
)
print(check_point.keys())
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict(file) -> str:
    image = Image.open(file).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim = 1)
        
    index = pred.item()
    
    return classes[index]
    
    