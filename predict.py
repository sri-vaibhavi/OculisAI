# --- IMPORTS ---
import os

import torch
import torch.nn as nn

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "effnet.pth")
IMG_SIZE = 300 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL LOADING (Runs once when app starts) ---
num_classes = 5 
model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- PREPROCESSING ---
def ben_graham_processing(img, img_size=300):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size))
    # Standard Ben Graham's processing for DR images
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), img_size / 30), -4, 128)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# --- INFERENCE FUNCTION ---
def get_prediction(image_path):
    orig_img = Image.open(image_path).convert("RGB")
    processed_img = ben_graham_processing(orig_img, IMG_SIZE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(processed_img).unsqueeze(0).to(DEVICE)
    
    labels = {
        0: "Mild Retinopathy",
        1: "Moderate Retinopathy",
        2: "No Diabetic Retinopathy",
        3: "Proliferative Retinopathy",
        4: "Severe Retinopathy"
    }
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)
    
    prediction_id = int(pred.item())
    dr_level_name = labels.get(prediction_id, "Unknown")
    confidence_pct = round(float(conf.item()) * 100, 2)
    return dr_level_name, confidence_pct, processed_img