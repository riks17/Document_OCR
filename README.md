# DocOCR Project - Local Setup Guide

## 1. Overview

This project consists of two services that must be run concurrently:
*   **`spring_backend` (Spring Boot):** The backend REST API. Runs on port `8080`.
*   **`ocr_service` (Python/FastAPI):** The microservice for OCR. Runs on port `8000`.

## 2. Prerequisites

Ensure the following are installed and available in your system's PATH:
*   Java JDK 17
*   Apache Maven (for dependency management within IntelliJ)
*   MySQL Server 8.0
*   Python 3.10
*   **Tesseract OCR Engine:** Install from the [official source](https://github.com/tesseract-ocr/tessdoc#installing-tesseract) and ensure the installer adds it to your system PATH.
    *   Verify with `tesseract --version` in a new terminal.

## 3. Configuration

### Step 3.1: Database

1.  Create a MySQL database named `identity_ocr_db`.
    ```sql
    CREATE DATABASE identity_ocr_db;
    ```
2.  Navigate to `spring_backend/src/main/resources/application.properties`.
3.  Update `spring.datasource.password` to match your local MySQL root/user password.

### Step 3.2: Tesseract (Python Service)

1.  Navigate to `ocr_service/app/ocr_utils.py`.
2.  If Tesseract was not added to your system PATH, you must uncomment and set the absolute path to your `tesseract.exe`:
    ```python
    # pytesseract.pytesseract.tesseract_cmd = r'C:\path\to\your\Tesseract-OCR\tesseract.exe'
    ```

## 4. Running the Application

The backend and OCR service must be run in parallel.

### IDE: Start the Backend (Spring Boot)

1.  Open the `spring_backend` directory as a new project in IntelliJ IDEA Ultimate.
2.  Wait for IntelliJ to index the project and resolve the Maven dependencies (a progress bar will appear at the bottom).
3.  Locate the main application file: `src/main/kotlin/com/example/docprocessor/DocprocessorApplication.kt`.
4.  Right-click the file and select **Run 'DocprocessorApplicationKt'** from the context menu.
5.  On the first startup, a default **SuperAdmin** credential set will be printed to the console. Note these for testing. The API will be available at `http://localhost:8080`.

### Terminal: Start the OCR Service (Python)

1.  Open a new terminal and navigate to the `ocr_service` root directory.
2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the FastAPI server:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

## 5. API Testing with Swagger UI

1.  Open the Swagger UI in your browser: [http://localhost:8080/swagger-ui.html](http://localhost:8080/swagger-ui.html)
2.  **Authenticate:**
    *   Use the `POST /api/auth/login` endpoint with the SuperAdmin credentials from the startup log to get a JWT token.
    *   Click the **"Authorize"** button (top right) and enter the token in the format: `Bearer <your_token>`.
3.  **Test Upload:**
    *   Use the `POST /api/documents/upload` endpoint.
    *   Click "Try it out", choose a file, and click "Execute".
    *   A `200 OK` response with the extracted JSON data confirms the entire pipeline is working.