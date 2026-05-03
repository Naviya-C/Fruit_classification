import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights, efficientnet_b0


class ResNet():
    def get_resnet_model50(num_classes):
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        
        for layer in model.parameters():
            layer.requires_grad = False

        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    def get_resnet_model18(num_class):
        model = resnet18(weights = ResNet18_Weights.DEFAULT)
        
        for layer in model.parameters():
            layer.requires_grad = False

        # Replace final layer        
        model.fc = nn.Linear(model.fc.in_features, num_class)

            
        return model
    
    
class EfficientNet():
    def get_efficientnet_b0(num_classes):
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        return model

