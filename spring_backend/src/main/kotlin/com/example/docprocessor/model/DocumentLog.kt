package com.example.docprocessor.model

import jakarta.persistence.*
import java.time.LocalDateTime

// Enum representing the types of documents supported by the system.
enum class DocumentType {
    PAN_CARD,
    VOTER_ID,
    PASSPORT,
    UNKNOWN
}

// Enum representing the processing status of a document.
enum class DocumentStatus {
    PENDING_VERIFICATION,
    REJECTED,
    UPLOADED,
    VERIFIED,
    FAILED_OCR,
    FAILED_STORAGE_MOVE
}

// Entity to log every document upload transaction for auditing and tracking purposes.
@Entity
@Table(name = "document_log")
data class DocumentLog(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Long? = null,
    val uploader: String,
    val fileName: String,
    @Enumerated(EnumType.STRING)
    val docType: DocumentType,
    @Enumerated(EnumType.STRING)
    val status: DocumentStatus,
    val createdAt: LocalDateTime = LocalDateTime.now()
)