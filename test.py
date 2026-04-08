import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
# Use absolute paths or paths relative to your workspace root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "effnet.pth")
IMAGE_PATH = "Test_dr_images/severe2.png"

IMG_SIZE = 300 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ PREPROCESSING ------------------
def ben_graham_processing(img, img_size=300):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size))
    
    # Highlights microaneurysms and hemorrhages
    # 
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), img_size / 30), -4, 128)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ------------------ LOAD MODEL ------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

num_classes = 5 
# Load structure
model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=num_classes)
# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------ INFERENCE ------------------
def predict_dr(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    orig_img = Image.open(image_path).convert("RGB")
    processed_img = ben_graham_processing(orig_img, IMG_SIZE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(processed_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    return orig_img, processed_img, prediction.item(), confidence.item()

# ------------------ EXECUTION ------------------

class_map = {2: "No DR", 0: "Mild", 1: "Moderate", 4: "Severe", 3: "Proliferative"}

try:
    orig, prep, pred_idx, conf = predict_dr(IMAGE_PATH)

    print(f"\n--- Prediction Result ---")
    print(f"Class: {class_map[pred_idx]}")
    print(f"Confidence: {conf*100:.2f}%")

    # Visualization
    plt.figure("DR Inference Result", figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig)
    plt.title("Original Fundus Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prep)
    plt.title(f"Processed - Pred: {class_map[pred_idx]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show() # In VS Code, this opens a window or shows in the Interactive tab

except Exception as e:
    print(f"Error occurred: {e}")