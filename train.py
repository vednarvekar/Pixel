"""
Improved training script with best practices - Fixed & Ready
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from pathlib import Path

# -----------------------
# Config
# -----------------------
BATCH_SIZE = 16
EPOCHS = 8 
LR = 0.001
WEIGHT_DECAY = 1e-4 
# This will default to CPU since you are using Intel Iris Xe
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

DATA_DIR = "dataset" # Ensure your folder is named 'dataset' in the same directory
MODELS_DIR = "models"
Path(MODELS_DIR).mkdir(exist_ok=True)

# Early stopping settings
PATIENCE = 3
best_val_acc = 0.0
patience_counter = 0

print(f"🖥️ Using device: {DEVICE}")
print(f"📁 Data directory: {DATA_DIR}")

# -----------------------
# Transforms
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------
# Datasets
# -----------------------
# Note: Ensure you have 'dataset/train' and 'dataset/val' folders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

print(f"\n📊 Dataset Loaded:")
print(f" Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
print(f" Classes: {train_dataset.classes}\n")

# -----------------------
# Model Setup
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze backbone (don't train the early layers)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 2 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(DEVICE)

# -----------------------
# Loss + Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# -----------------------
# Training Loop
# -----------------------
print(f"{'='*30}\n🚀 Starting Training\n{'='*30}")

for epoch in range(EPOCHS):
    # ---- Training Phase ----
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})

    epoch_train_loss = train_loss / len(train_dataset)
    epoch_train_acc = correct / len(train_dataset)

    # ---- Validation Phase ----
    model.eval()
    val_loss, val_correct = 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_correct / len(val_dataset)

    # Save History
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)

    print(f"Result: Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}%")

    # ---- Checkpointing & Early Stopping ----
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
        print(f"⭐ New Best Model Saved!")
    else:
        patience_counter += 1
        print(f"⏸️ No improvement for {patience_counter}/{PATIENCE} epochs.")

    if patience_counter >= PATIENCE:
        print("⏹️ Early stopping triggered.")
        break

# -----------------------
# Final Save
# -----------------------
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "final_model.pth"))
with open(os.path.join(MODELS_DIR, "history.json"), "w") as f:
    json.dump(history, f)

print(f"\n✅ Done! Best Val Accuracy: {best_val_acc*100:.2f}%")

















# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models 
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import os

# # CONFIG
# BATCH_SIZE = 16
# EPOCHS = 6
# LR = 0.001
# DEVICE = torch.device("cpu")

# DATA_DIR = "dataset"


# # TRANSFORMS
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])


# # DATASETS
# train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
# val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# # MODEL
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


# # FREEZE BACKBONE
# for param in model.parameters():
#     param.requires_grad = False


# # REPLACE FINAL LAYER
# model.fc = nn.Linear(model.fc.in_features, 2)
# model = model.to(DEVICE)


# # Loss + Optimizer
# criterion = nn.CrossEntropyLoss()
# optimiser = optim.Adam(model.fc.parameters(), lr=LR)


# # Training Loop
# for epoch in range(EPOCHS):
#     model.train()
#     train_loss = 0
#     correct = 0

#     for images, lables in tqdm(train_loader):
#         images, lables = images.to(DEVICE), lables.to(DEVICE)

#         outputs = model(images)
#         loss = criterion(outputs, lables)

#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()

#         train_loss += loss.item()
#         _, predicted  = torch.max(outputs, 1)
#         correct += (predicted == lables).sum().item()

#     train_acc = correct / len(train_dataset)
    
#     # VALIDATION 
#     model.eval()
#     val_correct = 0

#     with torch.no_grad():
#         for images, lables, in val_loader:
#             images, lables = images.to(DEVICE), lables.to(DEVICE)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             val_correct += (predicted == lables).sum().item()

#     val_acc = val_correct / len(val_dataset)


#     print(f"\nEpoch {epoch+1}/{EPOCHS}")
#     print(f"Train Accuracy: {train_acc:-4f}")
#     print(f,"Val Accuracy: {val_acc:-4f}")


# # SAVE MODEL
# torch.save(model.state_dict(), "pixel_model.pth")
# print("\nModel saved as pixel_model.pth")