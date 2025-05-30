import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ---------------------------
# ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser(description='Train a flower classifier')
parser.add_argument('data_dir', type=str, help='Path to dataset (e.g., flowers/)')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoint')
parser.add_argument('--arch', type=str, default='mobilenet_v2', help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# ---------------------------
# DEVICE SETUP
# ---------------------------
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

# ---------------------------
# DATA LOADERS
# ---------------------------
train_dir = os.path.join(args.data_dir, 'train')
valid_dir = os.path.join(args.data_dir, 'valid')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64)

# ---------------------------
# MODEL SETUP
# ---------------------------
model = models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, args.hidden_units),
    nn.ReLU(),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

model.to(device)

# ---------------------------
# TRAINING
# ---------------------------
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

print("Training started...")
for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            val_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Train loss: {running_loss/len(train_loader):.3f}.. "
          f"Validation loss: {val_loss/len(valid_loader):.3f}.. "
          f"Accuracy: {accuracy/len(valid_loader):.3f}")

# ---------------------------
# SAVE CHECKPOINT
# ---------------------------
model.class_to_idx = train_dataset.class_to_idx

checkpoint = {
    'arch': args.arch,
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict(),
    'classifier': model.classifier
}

torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
print("Checkpoint saved successfully.")
