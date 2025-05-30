import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser(description='Predict flower name from an image')
parser.add_argument('image_path', type=str, help='Path to image')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='JSON file mapping categories to names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

args = parser.parse_args()

# ----------------------------
# Device setup
# ----------------------------
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

# ----------------------------
# Load checkpoint
# ----------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

model = load_checkpoint(args.checkpoint)
model.to(device)
model.eval()

# ----------------------------
# Process image
# ----------------------------
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Resize
    if image.size[0] < image.size[1]:
        image.thumbnail((256, image.size[1]))
    else:
        image.thumbnail((image.size[0], 256))
    
    # Crop center
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert to np and normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.tensor(np_image).unsqueeze(0).float()

# ----------------------------
# Prediction
# ----------------------------
image = process_image(args.image_path)
image = image.to(device)

with torch.no_grad():
    output = model(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(args.top_k, dim=1)

# Map classes to labels
top_p = top_p.squeeze().tolist()
top_class = top_class.squeeze().tolist()

# Convert index to class
idx_to_class = {v: k for k, v in model.class_to_idx.items()}
top_labels = [idx_to_class[c] for c in top_class]

# Convert class to names (if JSON provided)
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

flower_names = [cat_to_name[str(label)] for label in top_labels]

# ----------------------------
# Print predictions
# ----------------------------
for i in range(len(flower_names)):
    print(f"{flower_names[i]}: {top_p[i]*100:.2f}%")
