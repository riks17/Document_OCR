package com.example.docprocessor.api

import com.example.docprocessor.service.DocumentProcessingService
import io.swagger.v3.oas.annotations.Operation
import io.swagger.v3.oas.annotations.security.SecurityRequirement
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.security.core.userdetails.UserDetails
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile

@CrossOrigin(origins = ["*"], maxAge = 3600)
@RestController
@RequestMapping("/api/documents")
@SecurityRequirement(name = "bearerAuth") // Applies the lock icon to all endpoints in this controller
class DocumentController(
    private val documentProcessingService: DocumentProcessingService
) {
    private val logger = LoggerFactory.getLogger(DocumentController::class.java)

    // Accepts a single document file for OCR processing and storage.
    @PostMapping("/upload", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    @PreAuthorize("hasAnyRole('USER', 'ADMIN', 'SUPERADMIN')")
    @Operation(summary = "Upload and process a single document")
    fun uploadAndProcessDocument(
        @RequestPart("file") file: MultipartFile,
        @AuthenticationPrincipal userDetails: UserDetails
    ): ResponseEntity<*> {
        if (file.isEmpty) {
            return ResponseEntity.badRequest().body(mapOf("message" to "File cannot be empty"))
        }
        try {
            logger.info("User '${userDetails.username}' attempting to upload file: '${file.originalFilename}'")
            val result = documentProcessingService.processUploadedDocument(file, userDetails.username)
            return ResponseEntity.ok(result)
        } catch (e: Exception) {
            logger.error("Error processing file for user '${userDetails.username}': ${e.message}", e)
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(mapOf("message" to "Error processing document: ${e.message}"))
        }
    }

    // Accepts multiple document files for batch processing.
    // Each file is processed individually, and a summary of results is returned.
    @PostMapping("/upload-bulk", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    @PreAuthorize("hasAnyRole('USER', 'ADMIN', 'SUPERADMIN')")
    @Operation(summary = "Upload and process multiple documents in bulk")
    fun uploadAndProcessDocumentsBulk(
        // ✅ CORRECTED: Using @RequestPart for the list of files. This is the key fix.
        @RequestPart("files") files: List<MultipartFile>,
        @AuthenticationPrincipal userDetails: UserDetails
    ): ResponseEntity<*> {
        logger.info("Received bulk upload request with ${files.size} files by user: ${userDetails.username}")
        if (files.isEmpty()) {
            return ResponseEntity.badRequest().body(mapOf("message" to "No files provided for bulk upload"))
        }

        val results = mutableListOf<Map<String, Any?>>()
        val username = userDetails.username

        files.forEach { file ->
            val originalFilename = file.originalFilename ?: "unknown_file_${System.currentTimeMillis()}"
            if (file.isEmpty) {
                logger.warn("Skipping empty file '$originalFilename' in bulk upload for user: $username")
                results.add(
                    mapOf(
                        "fileName" to originalFilename,
                        "status" to "SKIPPED",
                        "message" to "File is empty"
                    )
                )
            } else {
                try {
                    logger.info("Processing file '$originalFilename' in bulk upload for user: $username")
                    // Calling the single-file processing logic from the service.
                    val resultData = documentProcessingService.processUploadedDocument(file, username)
                    results.add(
                        mapOf(
                            "fileName" to originalFilename,
                            "status" to "SUCCESS",
                            "data" to resultData
                        )
                    )
                    logger.info("Successfully processed file '$originalFilename' for user: $username")
                } catch (e: Exception) {
                    logger.error("Error processing file '$originalFilename' for user $username: ${e.message}", e)
                    results.add(
                        mapOf(
                            "fileName" to originalFilename,
                            "status" to "FAILURE",
                            "error" to e.message
                        )
                    )
                }
            }
        }
        logger.info("Finished bulk upload processing for user: $username. Processed ${results.size} entries.")
        return ResponseEntity.ok(results)
    }


    // Retrieves processed PAN card data by its database ID.
    // Accessible only by Admin and SuperAdmin roles.
    @GetMapping("/pan/{id}")
    @PreAuthorize("hasAnyRole('ADMIN', 'SUPERADMIN')")
    @Operation(summary = "Get PAN card data by ID", security = [SecurityRequirement(name = "bearerAuth")])
    fun getPanDataById(@PathVariable id: Long): ResponseEntity<*> {
        val panData = documentProcessingService.getPanDataById(id)
        return if (panData != null) {
            ResponseEntity.ok(panData)
        } else {
            ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(mapOf("message" to "PAN Card data with ID $id not found"))
        }
    }

    // Retrieves processed Voter ID data by its database ID.
    // Accessible only by Admin and SuperAdmin roles.
    @GetMapping("/voter-id/{id}")
    @PreAuthorize("hasAnyRole('ADMIN', 'SUPERADMIN')")
    @Operation(summary = "Get Voter ID data by ID", security = [SecurityRequirement(name = "bearerAuth")])
    fun getVoterIdDataById(@PathVariable id: Long): ResponseEntity<*> {
        val voterIdData = documentProcessingService.getVoterIdDataById(id)
        return if (voterIdData != null) {
            ResponseEntity.ok(voterIdData)
        } else {
            ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(mapOf("message" to "Voter ID data with ID $id not found"))
        }
    }

    // Retrieves processed Passport data by its database ID.
    // Accessible only by Admin and SuperAdmin roles.
    @GetMapping("/passport/{id}")
    @PreAuthorize("hasAnyRole('ADMIN', 'SUPERADMIN')")
    @Operation(summary = "Get Passport data by ID", security = [SecurityRequirement(name = "bearerAuth")])
    fun getPassportDataById(@PathVariable id: Long): ResponseEntity<*> {
        val passportData = documentProcessingService.getPassportDataById(id)
        return if (passportData != null) {
            ResponseEntity.ok(passportData)
        } else {
            ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(mapOf("message" to "Passport data with ID $id not found"))
        }
    }
}