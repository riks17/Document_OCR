"""
This module provides utilities for performing OCR with Tesseract.
It includes functions for image preprocessing and for running Tesseract with configurations optimized for different document types and fields.
"""
import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
from datetime import datetime
import os

# Tesseract OCR path (if required)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define processed_dir at the module level for the helper function to use
processed_dir = "processed_images" 
os.makedirs(processed_dir, exist_ok=True)

def preprocess_image(image_path, save_path=None):
    """Applies a general-purpose preprocessing pipeline to an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    height, width = thresh.shape
    resized = cv2.resize(thresh, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR_EXACT)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)

    if save_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(processed_dir, f"{base_name}_preprocessed.png")
    cv2.imwrite(save_path, sharpened)
    return save_path

def _extract_text_from_single_image(image_path, doc_type=None, field=None, skip_preprocessing=False):
    """
    Internal helper to preprocess, configure OCR, and extract text from a single image path.
    Can optionally skip the preprocessing step.
    """
    if doc_type in ("voterid_new", "voterid_old") or skip_preprocessing:
        processed_image_path = image_path
        if skip_preprocessing:
            print(f"Skipping preprocessing for field '{field}' and using raw image.")
    else:
        processed_image_path = preprocess_image(image_path)
    
    # Step 2: Select the correct Tesseract configuration
    if doc_type == 'passport':
        if field in ['dob', 'expiry']:  
            custom_config = r'--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789/'
        elif field == 'passport_number':
            custom_config = r'--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$'
        elif field in ['name', 'surname']:
            custom_config = r'--oem 3 --psm 7 -l eng'
        elif field == 'gender':
            custom_config = r'--psm 7 --oem 1 -c tessedit_char_whitelist=M/F'
        else:
             custom_config = r'--oem 1 --psm 7 -l eng'
    elif doc_type == 'pan':
        if field == 'pan':
            custom_config = r'--oem 1 --psm 13 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        elif field == 'dob':
            custom_config = r'--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789/'
        else:
            custom_config = r'--oem 1 --psm 7 -l eng'
    else: # Default/VoterID config
        if field == 'voter_id':
            custom_config = r'--oem 1 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
        else:
            custom_config = r'--oem 1 --psm 7 -l eng -c preserve_interword_spaces=1'

    # Step 3: Run OCR
    try:
        text = pytesseract.image_to_string(Image.open(processed_image_path), config=custom_config)
    except FileNotFoundError:
        print(f"Warning: Image path not found for OCR: {processed_image_path}")
        return "" 
    
    # Step 4: Apply field-specific post-processing
    if doc_type in ("voterid_new", "voterid_old") and field == 'name':
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    return text.strip()

def extract_text(image_path_or_list, doc_type=None, field=None, skip_preprocessing=False):
    """
    Extracts text from a single image path or a list of image paths.
    Can optionally skip preprocessing.
    """
    if isinstance(image_path_or_list, list):
        return [
            _extract_text_from_single_image(img_path, doc_type=doc_type, field=field, skip_preprocessing=skip_preprocessing)
            for img_path in image_path_or_list
        ]
    else:
        return _extract_text_from_single_image(image_path_or_list, doc_type=doc_type, field=field, skip_preprocessing=skip_preprocessing)