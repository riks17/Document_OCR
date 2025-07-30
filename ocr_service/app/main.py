from fastapi import FastAPI, UploadFile, File
from app.classifier import classify_document
from app.bbox_predictor import run_bbox_model
from app.cropper import run_pre_classification_cropping
from app.ocr_utils import extract_text
from app import corrections

import shutil
import os
import re

app = FastAPI()

# Ensure necessary directories exist when the application starts.
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed_images", exist_ok=True)

# --- HELPER FUNCTION FOR API RESPONSE ---
def create_error_response(filename, message, ocr_results=None):
    """Helper function to create consistent error responses."""
    response = {"filename": filename, "document_type": "unknown", "message": message}
    if ocr_results:
        response["ocr_results"] = ocr_results
    return response

# --- MAIN API ENDPOINT ---
@app.post("/ocr/process/")
async def predict(files: list[UploadFile] = File(...)):
    # Processes a batch of uploaded document images.
    results = []
    for file in files:
        image_path = f"uploads/{file.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # --- STAGE 1: Pre-classification Cropping ---
            run_pre_classification_cropping(image_path)

            # --- STAGE 2: Document Classification ---
            doc_type = classify_document(image_path)
            print(f"Predicted document type for {file.filename}: {doc_type}")

            # --- STAGE 3: Field Detection ---
            cropped_image_paths = run_bbox_model(doc_type, image_path)

            # --- STAGE 4: Targeted OCR and Correction (Logic varies by doc_type) ---
            if doc_type == "pan":
                pan_fields = ["dob", "father", "name", "pan"]
                ocr_results = {}
                for field in pan_fields:
                    crop_path = cropped_image_paths.get(field)
                    if not crop_path: continue

                    # If initial OCR is empty, try again without preprocessing.
                    text = extract_text(crop_path, doc_type=doc_type, field=field)
                    if not text.strip():
                        print(f"Initial OCR for PAN field '{field}' was empty. Retrying...")
                        text = extract_text(crop_path, doc_type=doc_type, field=field, skip_preprocessing=True)
                    
                    # Apply corrections
                    if field == 'dob': text = corrections.correct_date_string(text)
                    ocr_results[field] = text

                # Post-process and validate all extracted fields together
                if 'pan' in ocr_results:
                    cleaned_pan = corrections.clean_id_text(ocr_results.get("pan", ""))
                    ocr_results['pan'] = corrections.apply_pan_corrections(cleaned_pan)
                for field in ['name', 'father']:
                    if field in ocr_results: ocr_results[field] = corrections.clean_name_field(ocr_results[field])

                # Final validation check
                if not bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", ocr_results.get("pan", ""))):
                    results.append(create_error_response(file.filename, "Document type could not be verified (invalid PAN format).", ocr_results))
                    continue
                results.append({"filename": file.filename, "document_type": doc_type, "ocr_results": ocr_results})
            
            elif doc_type == "passport":
                passport_fields = ["dob", "expiry", "surname", "gender", "name", "passport_number"]
                ocr_results = {}
                
                for field in passport_fields:
                    crop_path = cropped_image_paths.get(field)
                    if not crop_path: continue

                    # --- Attempt 1: Using the pre-processed image ---
                    text_attempt1 = extract_text(crop_path, doc_type='passport', field=field)
                    
                    if field in ['dob', 'expiry']:
                        processed_text1 = corrections.correct_date_string(text_attempt1)
                    elif field == 'passport_number':
                        processed_text1 = corrections.apply_passport_corrections(text_attempt1)
                    else:
                        processed_text1 = text_attempt1

                    # --- Validate the result of the first attempt ---
                    is_valid = True
                    if not text_attempt1.strip():
                        is_valid = False
                        print(f"Initial OCR for passport field '{field}' was empty.")
                    elif field in ['dob', 'expiry']:
                        if not corrections.is_valid_date_format(processed_text1):
                            is_valid = False
                            print(f"Attempt 1 for '{field}' ('{processed_text1}') failed format validation.")
                    elif field == 'passport_number':
                        if not corrections.is_valid_passport_format(processed_text1):
                            is_valid = False
                            print(f"Attempt 1 for '{field}' ('{processed_text1}') failed format validation.")

                    # --- Decide: Use Attempt 1's result OR retry with Attempt 2? ---
                    if is_valid:
                        # First attempt was good, store its result.
                        ocr_results[field.lower()] = processed_text1
                    else:
                        # First attempt failed, so we retry on the raw (non-pre-processed) image.
                        print(f"Retrying field '{field}' on raw image...")
                        text_attempt2 = extract_text(crop_path, doc_type='passport', field=field, skip_preprocessing=True)
                        
                        # Process the result of the second attempt.
                        if field in ['dob', 'expiry']:
                            processed_text2 = corrections.correct_date_string(text_attempt2)
                        elif field == 'passport_number':
                            processed_text2 = corrections.apply_passport_corrections(text_attempt2)
                        else:
                            processed_text2 = text_attempt2
                        
                        # Store the result of the second attempt, whatever it may be.
                        ocr_results[field.lower()] = processed_text2

                # Combine name and surname into a single 'name' field for a cleaner API response.
                name_part = corrections.clean_name_field(ocr_results.get('name', ''))
                surname_part = corrections.clean_name_field(ocr_results.get('surname', ''))
                full_name_parts = [part for part in [name_part, surname_part] if part]
                ocr_results['name'] = " ".join(full_name_parts)
                if 'surname' in ocr_results:
                    del ocr_results['surname']

                # Standardize gender output
                if 'gender' in ocr_results:
                    if 'F' in ocr_results.get('gender', '').upper():
                        ocr_results['gender'] = 'Female'
                    else:
                        ocr_results['gender'] = 'Male'
                
                # Final validation on the stored, corrected number
                final_passport_num = ocr_results.get('passport_number', '')
                if not corrections.is_valid_passport_format(final_passport_num):
                    results.append(create_error_response(file.filename, "Invalid Passport Number format after all checks.", ocr_results))
                    continue

                results.append({"filename": file.filename, "document_type": doc_type, "ocr_results": ocr_results})

            elif doc_type in ["voterid_new", "voterid_old"]:
                voterid_fields = ["date", "gender", "name", "voter_id"]
                ocr_results = {}
                for field in voterid_fields:
                    crop_path = cropped_image_paths.get(field)
                    if not crop_path: continue
                    text = extract_text(crop_path, doc_type=doc_type, field=field)
                    if field == 'date': text = corrections.correct_date_string(text)
                    ocr_results[field] = text

                # Apply post-processing and corrections
                if 'name' in ocr_results: 
                    ocr_results['name'] = corrections.clean_name_field(ocr_results['name'])
                if 'gender' in ocr_results:
                    if 'F' in ocr_results.get('gender', '').upper(): ocr_results['gender'] = 'Female'
                    else: ocr_results['gender'] = 'Male'
                if 'voter_id' not in ocr_results or not ocr_results.get('voter_id', '').strip():
                    results.append(create_error_response(file.filename, "Voter ID number not found.", ocr_results))
                    continue
                cleaned_voter_id = corrections.clean_id_text(ocr_results.get("voter_id", ""))
                final_voter_id = corrections.correct_and_reformat_voter_id(cleaned_voter_id)
                ocr_results["voter_id"] = final_voter_id

                # Final validation for both old and new Voter ID formats
                is_valid_voterid = bool(re.match(r"^[A-Z]{3}/[0-9]{7}$", final_voter_id) or re.match(r"^[A-Z]{2}/[0-9]{2}/[0-9]{3}/[0-9]{6}$", final_voter_id) or re.match(r"^[A-Z]{3}[0-9]{7}$", final_voter_id))
                if not is_valid_voterid:
                    results.append(create_error_response(file.filename, "Invalid Voter ID format.", ocr_results))
                    continue
                results.append({"filename": file.filename, "document_type": doc_type, "ocr_results": ocr_results})
            
            else:
                results.append({"filename": file.filename, "document_type": doc_type, "message": "Handling for this document type is not yet implemented."})

        except ValueError as e:
            results.append(create_error_response(file.filename, str(e)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"filename": file.filename, "document_type": "unknown", "error": f"An unexpected error occurred: {e}"})

    return {"results": results}