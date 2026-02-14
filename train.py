import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models 
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# CONFIG
BATCH_SIZE = 16
EPOCHS = 6
LR = 0.001
DEVICE = torch.device("cpu")

DATA_DIR = "dataset"


# TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# DATASETS
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)