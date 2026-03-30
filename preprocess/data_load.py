from torchvision import datasets
from torch.utils.data import DataLoader
from preprocess.preprocessing import data_transform
import os


train_data = datasets.ImageFolder("/home/naviya-c/Desktop/Projects/fruit_classification/dataset/fruits-360_original-size/fruits-360-original-size/Training", transform = data_transform['train_transform'])
val_data = datasets.ImageFolder("/home/naviya-c/Desktop/Projects/fruit_classification/dataset/fruits-360_original-size/fruits-360-original-size/Validation", transform = data_transform['test_transform'])
test_data = datasets.ImageFolder("/home/naviya-c/Desktop/Projects/fruit_classification/dataset/fruits-360_original-size/fruits-360-original-size/Test", transform = data_transform['test_transform'])

train_load = DataLoader(train_data, batch_size = 4, shuffle = True)
val_load = DataLoader(val_data, batch_size = 4)
test_load = DataLoader(test_data, batch_size = 4)

print("Train classes:", train_data.classes[:10])
print("Val classes:", val_data.classes[:10])

print("Train class count:", len(train_data.classes))
print("Val class count:", len(val_data.classes))

train_data.class_to_idx == val_data.class_to_idx

def num_class():
    return len(train_data.classes) 