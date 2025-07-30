"""
This module contains functions for cleaning, correcting, and validating text extracted via OCR. 
It includes generic cleaning functions and document-specific correction rules.
"""
import re

# --- CHARACTER CORRECTION MAPS ---
NUMERIC_CORRECTIONS = {
    'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'G': '6'
}
ALPHA_CORRECTIONS = {v: k for k, v in NUMERIC_CORRECTIONS.items() if k != 'I'}

# --- GENERIC CLEANING AND VALIDATION ---
def clean_id_text(text):
    # Removes all non-alphanumeric characters except $ from a string and converts to uppercase.
    return re.sub(r'[^A-Z0-9$/]', '', text.upper())

def clean_name_field(text):
    # Removes all non-alphabetic characters except for spaces from a string.
    if not isinstance(text, str):
        return ""
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def is_valid_date_format(date_string):
    # Checks if a string strictly matches the DD/MM/YYYY format.
    if not isinstance(date_string, str):
        return False
    return bool(re.match(r"^\d{2}/\d{2}/\d{4}$", date_string.strip()))

def is_valid_passport_format(passport_num):
    # Checks if a string matches the standard 1 Alpha, 7 Numeric passport format.
    if not isinstance(passport_num, str):
        return False
    return bool(re.match(r"^[A-Z][0-9]{7}$", passport_num))

def correct_date_string(date_str):
    # Corrects specific, common OCR errors in a DD/MM/YYYY date string.
    if not isinstance(date_str, str) or date_str.count('/') != 2:
        return date_str
    try:
        parts = date_str.strip().split('/')
        if len(parts) != 3: return date_str
        day_str, month_str, year_str = parts
        if len(day_str) == 2 and day_str.startswith('4'):
            day_str = '1' + day_str[1]
        if len(day_str) == 2 and day_str.startswith('9'):
            day_str = '2' + day_str[1]
        if len(day_str) == 2 and day_str.startswith('6'):
            day_str = '0' + day_str[1]
        if len(month_str) == 2 and month_str.startswith('4') and month_str[1] in '012':
            month_str = '1' + month_str[1]
        if len(year_str) == 4 and year_str.startswith('4'):
            year_str = '1' + year_str[1]
        if len(year_str) == 4 and year_str.startswith('7'):
            year_str = '1' + year_str[1]
        if len(year_str) == 4 and year_str.startswith('9'):
            year_str = '2' + year_str[1]
        return f"{day_str}/{month_str}/{year_str}"
    except Exception:
        return date_str

# --- DOCUMENT-SPECIFIC CORRECTIONS ---
def apply_pan_corrections(pan_number):
    # Corrects common OCR errors in a 10-digit PAN number based on its format.
    if len(pan_number) == 10:
        pan_list = list(pan_number)
        for i in range(5):
            if pan_list[i].isdigit() and pan_list[i] in ALPHA_CORRECTIONS:
                pan_list[i] = ALPHA_CORRECTIONS[pan_list[i]]
        for i in range(5, 9):
            if pan_list[i].isalpha() and pan_list[i] in NUMERIC_CORRECTIONS:
                pan_list[i] = NUMERIC_CORRECTIONS[pan_list[i]]
        if pan_list[9].isdigit() and pan_list[9] in ALPHA_CORRECTIONS:
            pan_list[9] = ALPHA_CORRECTIONS[pan_list[9]]
        return ''.join(pan_list)
    return pan_number

def apply_passport_corrections(raw_text):
    # Cleans and corrects a raw OCR string for a passport number in the correct order.
    if not isinstance(raw_text, str):
        return ""
    
    temp_text = raw_text.strip()

    cleaned_text = clean_id_text(temp_text)

    if cleaned_text.startswith('$'):
        print(f"Applying passport correction: '$' -> 'S'")
        cleaned_text = 'S' + cleaned_text[1:]

    if len(cleaned_text) == 8:
        pass_list = list(cleaned_text)
        if pass_list[0].isdigit() and pass_list[0] in ALPHA_CORRECTIONS:
            pass_list[0] = ALPHA_CORRECTIONS[pass_list[0]]
        for i in range(1, 8):
            if pass_list[i].isalpha() and pass_list[i] in NUMERIC_CORRECTIONS:
                pass_list[i] = NUMERIC_CORRECTIONS[pass_list[i]]
        return ''.join(pass_list)
    
    return cleaned_text

def correct_and_reformat_voter_id(voter_id):
    # Corrects and reformats a Voter ID based on its known formats.
    temp = voter_id.replace('/', '')
    temp_id = list(temp)
    corrected_id = ""

    # Handle old 10-digit format (AAA1111111)
    if len(temp_id) == 10:
        for i in range(3):
            if temp_id[i].isdigit() and temp_id[i] in ALPHA_CORRECTIONS:
                temp_id[i] = ALPHA_CORRECTIONS[temp_id[i]]
        for i in range(3, 10):
            if temp_id[i].isalpha() and temp_id[i] in NUMERIC_CORRECTIONS:
                temp_id[i] = NUMERIC_CORRECTIONS[temp_id[i]]
        corrected_id = ''.join(temp_id)
    # Handle new 13-digit format (AA/11/111/111111)
    elif len(temp_id) == 13:
        for i in range(2):
            if temp_id[i].isdigit() and temp_id[i] in ALPHA_CORRECTIONS:
                temp_id[i] = ALPHA_CORRECTIONS[temp_id[i]]
        for i in range(2, 13):
            if temp_id[i].isalpha() and temp_id[i] in NUMERIC_CORRECTIONS:
                temp_id[i] = NUMERIC_CORRECTIONS[temp_id[i]]
        corrected_id = ''.join(temp_id)
    else:
        # If length is wrong, return original.
        return voter_id
    
    # Re-apply the separator if it was present in the original OCR text.
    has_separator = '/' in voter_id
    if has_separator:
        if len(corrected_id) == 13:
            return f"{corrected_id[:2]}/{corrected_id[2:4]}/{corrected_id[4:7]}/{corrected_id[7:]}"
        elif len(corrected_id) == 10:
            return f"{corrected_id[:3]}/{corrected_id[3:]}"
        
    return corrected_id