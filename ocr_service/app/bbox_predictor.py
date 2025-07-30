# This module detects specific fields (e.g., name, DOB) on a classified document.
# It also handles document orientation correction before field detection.
# It uses document-specific YOLO models to find bounding boxes for each field.

import cv2
import os
import torchvision.transforms as T
from ultralytics import YOLO
import numpy as np
import uuid
import pytesseract
import re

# --- HELPER FUNCTIONS ---
def expand_box(x1, y1, x2, y2, img_shape, margin_ratio=0.05):
    """
    Expands a bounding box by a percentage of its size, clamping to image boundaries.
    This helps to include text that might be slightly outside the detected box.
    """
    h, w = img_shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    margin_x, margin_y = int(box_w * margin_ratio), int(box_h * margin_ratio)
    
    new_x1 = max(0, x1 - margin_x)
    new_y1 = max(0, y1 - margin_y)
    new_x2 = min(w, x2 + margin_x)
    new_y2 = min(h, y2 + margin_y)

    return new_x1, new_y1, new_x2, new_y2

def _get_preprocessed_image_data(image_data):
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    return gray

# --- PERFORMS OCR ON THE IMAGE AND RETURNS THE AVERAGE CONFIDENCE SCORE ---
def get_ocr_confidence(image):
    custom_config = r'--oem 3 --psm 6'
    try:
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        confs = [int(conf) for conf in data["conf"] if conf != '-1']
        avg_conf = sum(confs) / len(confs) if confs else 0
        return avg_conf
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not in your PATH. OCR functionality will not work.")
        return 0
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return 0

# --- FIELD DETECTION AND ORIENTATION ---
def run_bbox_model(doc_type, img_path):
    """
    Runs field detection and orientation on an ALREADY CROPPED document image.
    Loads the model based on the document type (e.g., 'pan_model.pt', 'passport_model.pt').
    """
    # Mapping from YOLO class names to expected field names for passport
    YOLO_FIELD_MAPPING = {
        "DOB": "dob",
        "ExpiryDate": "expiry",
        "Surname": "surname",
        "gender": "gender",
        "name": "name",
        "passport number": "passport_number"
    } if doc_type == "passport" else {}

    field_model_path = f"models/{doc_type}_model.pt"
    if not os.path.exists(field_model_path):
        raise FileNotFoundError(f"Field detection model not found at '{field_model_path}'.")

    field_model = YOLO(field_model_path)
    processed_dir = "processed_images"
    os.makedirs(processed_dir, exist_ok=True)

    img_for_processing = cv2.imread(img_path)
    if img_for_processing is None:
        raise ValueError(f"Failed to load pre-cropped image from path: {img_path}")

    img_cv = img_for_processing
    h, w = img_cv.shape[:2]

    # Orientation logic
    if doc_type in ["passport", "pan"]:
        if h > w:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
    elif doc_type in ["voterid_new", "voterid_old"]:
        if w > h:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)

    if doc_type in ["pan", "voterid_new"]:
        preprocessed_for_ocr = _get_preprocessed_image_data(img_cv)
        score_0 = get_ocr_confidence(preprocessed_for_ocr)
        img_180_preprocessed = cv2.rotate(preprocessed_for_ocr, cv2.ROTATE_180)
        score_180 = get_ocr_confidence(img_180_preprocessed)
    else:
        score_0 = get_ocr_confidence(img_cv)
        img_180 = cv2.rotate(img_cv, cv2.ROTATE_180)
        score_180 = get_ocr_confidence(img_180)

    if score_180 > score_0:
        img_cv = cv2.rotate(img_cv, cv2.ROTATE_180)

    cv2.imwrite(img_path, img_cv)

    # --- Field Detection ---
    results = field_model(img_cv, verbose=False)
    result = results[0]
    names = getattr(result, 'names', None)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    bbox_to_path = {}

    # Check if the model has OBB (Oriented Bounding Boxes) or regular boxes
    boxes = getattr(result.obb, 'xyxy', None)
    classes = getattr(result.obb, 'cls', None)
    is_obb = boxes is not None

    # If not an OBB model, fall back to standard horizontal boxes.
    if not is_obb:
        boxes = getattr(result.boxes, 'xyxy', None)
        classes = getattr(result.boxes, 'cls', None)

    if boxes is None or len(boxes) == 0:
        raise ValueError(f"No bounding boxes for fields detected for {doc_type}.")

    for i, (box_tensor, cls_idx) in enumerate(zip(boxes, classes)):
        box = box_tensor.cpu().numpy().astype(int)
        x1, y1, x2, y2 = box[:4]

        # Apply a small expansion to PAN card fields to ensure full capture.
        if doc_type == "pan":
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, img_cv.shape)

        cropped = img_cv[y1:y2, x1:x2]
        raw_class_name = names[int(cls_idx)] if names is not None else f"class_{int(cls_idx)}"
        mapped_class_name = YOLO_FIELD_MAPPING.get(raw_class_name, raw_class_name)

        final_path = process_cropped_image(
            cropped, base_name, mapped_class_name, output_dir=processed_dir
        )

        bbox_to_path[mapped_class_name] = final_path

    return bbox_to_path


# SAVES THE CROPPED IMAGE AND RETURNS THE PATH
def process_cropped_image(cropped, base_name, class_name, output_dir="processed_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    if cropped is None or cropped.size == 0:
        print(f"Warning: Cannot save empty image for class '{class_name}'. Skipping.")
        return f"path/to/empty_image_placeholder.jpg"
        
    final_image = cropped.copy()
    filename = f"{base_name}_final_crop_{class_name}.jpg"
    final_path = os.path.join(output_dir, filename)
    cv2.imwrite(final_path, final_image)
    
    return final_path