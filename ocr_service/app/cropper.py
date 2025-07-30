# This module is responsible for performing a generic, pre-classification crop to remove background noise from a document image using a universal cropping model. 

import os
import cv2
import torch
from ultralytics import YOLO

def run_pre_classification_cropping(img_path):
    cropping_model_path = "models/cropping_model.pt"
    if not os.path.exists(cropping_model_path):
        raise FileNotFoundError(f"Universal cropping model not found at '{cropping_model_path}'.")

    cropping_model = YOLO(cropping_model_path)
    
    img_to_crop = cv2.imread(img_path)
    if img_to_crop is None:
        raise ValueError(f"Failed to load image for cropping from path: {img_path}")

    print("Running pre-classification background cropping...")
    cropping_results = cropping_model(img_to_crop, verbose=False)
    
    detected_boxes = cropping_results[0].boxes.xyxy
    if len(detected_boxes) > 0:
        print("Document boundary detected. Cropping background...")
        # Find the bounding box with the largest area
        areas = (detected_boxes[:, 2] - detected_boxes[:, 0]) * (detected_boxes[:, 3] - detected_boxes[:, 1])
        best_box_idx = torch.argmax(areas)
        x1, y1, x2, y2 = detected_boxes[best_box_idx].cpu().numpy().astype(int)
        
        cropped_image = img_to_crop[y1:y2, x1:x2]
        
        # Overwrite the original image with the cropped version
        print(f"Overwriting '{img_path}' with cropped version for classification.")
        cv2.imwrite(img_path, cropped_image)
    else:
        print("WARNING: Document boundary not detected for pre-cropping. Proceeding with original image.")
    
    return