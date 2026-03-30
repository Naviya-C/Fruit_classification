import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

#Doing transfer learning. Therefore freeze all hidden layers but final layer changes to num of foods.
def transfer_model(num_class):
    model = resnet50(weights = ResNet50_Weights.DEFAULT)
    
    for layer in model.parameters():
        layer.requires_grad = False
        
    #Replace final layer 
    #for layer in model.layer4.parameters():
    #   layer.requires_grad = True
        
    model.fc = nn.Linear(model.fc.in_features, num_class) 
    
    return model