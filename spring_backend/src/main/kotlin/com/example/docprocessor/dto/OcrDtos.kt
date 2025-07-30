package com.example.docprocessor.dto

import com.fasterxml.jackson.annotation.JsonProperty

// Models the root JSON object received from the Python OCR service.
data class PythonOcrResponse(
    val results: List<PythonOcrResult>
)

// Models a single result object within the Python OCR response list.
data class PythonOcrResult(
    val filename: String,
    @JsonProperty("document_type")
    val documentType: String,
    @JsonProperty("ocr_results")
    val ocrResults: Map<String, String>? = null, // Can be null in case of an error
    val message: String? = null,
    val error: String? = null
)

// A generic DTO for a successful document processing response.
// Note: The actual saved entity is returned by the service, this is a potential alternative.
data class DocumentProcessingResult(
    val documentId: Long,
    val documentType: String,
    val message: String
)