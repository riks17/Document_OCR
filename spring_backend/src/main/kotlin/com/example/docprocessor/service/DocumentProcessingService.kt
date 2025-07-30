package com.example.docprocessor.service

import com.example.docprocessor.model.*
import com.example.docprocessor.repository.*
import com.example.docprocessor.service.client.OcrServiceClient
import org.apache.pdfbox.Loader
import org.apache.pdfbox.rendering.PDFRenderer
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.multipart.MultipartFile
import java.awt.image.BufferedImage
import java.io.File
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.*
import javax.imageio.ImageIO

// The core service responsible for the entire document processing workflow.
// It acts as an orchestrator, handling file storage, PDF conversion,
// communication with the external OCR service, and database persistence.
@Service
class DocumentProcessingService(
    private val ocrServiceClient: OcrServiceClient,
    private val panDataRepository: PanDataRepository,
    private val voterIdDataRepository: VoterIdDataRepository,
    private val passportDataRepository: PassportDataRepository,
    private val documentLogRepository: DocumentLogRepository
) {
    private val logger = LoggerFactory.getLogger(DocumentProcessingService::class.java)
    private val fileStorageLocation: Path = Paths.get("./storage").toAbsolutePath().normalize()
    private val tempFileLocation: Path = Paths.get("./temp").toAbsolutePath().normalize()
    private val timestampDirFormatter = DateTimeFormatter.ofPattern("yyyyMMdd")
    private val dateFormatter = DateTimeFormatter.ofPattern("dd/MM/yyyy")

    init {
        try {
            Files.createDirectories(fileStorageLocation)
            Files.createDirectories(tempFileLocation)
        } catch (ex: Exception) {
            throw RuntimeException("Could not create storage/temp directories.", ex)
        }
    }

    @Transactional
    fun processUploadedDocument(multipartFile: MultipartFile, username: String): Any {
        var tempStagedFile: File? = null
        var convertedImageFile: File? = null
        val processingTimestamp = LocalDateTime.now()

        try {
            // Step 1: Stage the uploaded file to a temporary location for safe processing.
            val (stagedFile, uniqueFilename) = storeFileToTemp(multipartFile)
            tempStagedFile = stagedFile
            var fileToProcess: File = tempStagedFile

            // Step 2: If the file is a PDF, convert it to an image for the OCR service.
            if (isPdf(multipartFile, tempStagedFile)) {
                logger.info("PDF file detected. Converting to image for OCR.")
                convertedImageFile = convertPdfToImage(tempStagedFile)
                fileToProcess = convertedImageFile
            }

            // Step 3: Call the external Python OCR service using the declarative Feign client.
            val multipartFileAdapter = FileToMultipartFileAdapter(fileToProcess)
            val pythonResponse = ocrServiceClient.processDocument(multipartFileAdapter)
            val ocrResult = pythonResponse.results.firstOrNull()
                ?: throw IllegalStateException("OCR service returned an empty result list.")

            // Step 4: Validate the response from the OCR service.
            if (ocrResult.ocrResults == null || ocrResult.message != null || ocrResult.error != null) {
                val errorMessage = ocrResult.message ?: ocrResult.error ?: "Unknown error from OCR service"
                throw IllegalStateException("OCR processing failed: $errorMessage")
            }

            // Step 5: After a successful OCR, move the original file to permanent, organized storage.
            val fieldsMap = ocrResult.ocrResults
            val docType = mapPythonDocTypeToEnum(ocrResult.documentType)
            if (docType == DocumentType.UNKNOWN) {
                throw IllegalArgumentException("Unsupported document type received: ${ocrResult.documentType}")
            }
            val finalRelativePath = moveFileToFinal(tempStagedFile, docType, processingTimestamp, uniqueFilename)
            tempStagedFile = null // Mark as moved so it's not deleted in the `finally` block.

            // Step 6: Save the extracted data to the appropriate database table.
            val storedFilePathForDb = finalRelativePath.toString().replace(File.separatorChar, '/')
            val result = saveData(docType, fieldsMap, storedFilePathForDb, username, processingTimestamp)
            logDocument(username, uniqueFilename, docType, DocumentStatus.VERIFIED)
            return result

        } catch (e: Exception) {
            logger.error("Document processing failed for user '$username': ${e.message}", e)
            logDocument(username, multipartFile.originalFilename ?: "unknown", DocumentType.UNKNOWN, DocumentStatus.REJECTED, e.message)
            throw e // Re-throw to be caught by the GlobalExceptionHandler
        } finally {
            // Step 7: Always clean up any temporary files that were created during the process.
            convertedImageFile?.delete()
            tempStagedFile?.delete()
            logger.info("Cleanup of temporary files complete for user '$username'.")
        }
    }

    private fun saveData(docType: DocumentType, fields: Map<String, String>, path: String, user: String, timestamp: LocalDateTime): Any {
        return when (docType) {
            DocumentType.PAN_CARD -> {
                val panNumber = fields["pan"] ?: throw IllegalArgumentException("PAN number is required and was not extracted.")
                panDataRepository.save(PanData(name = fields["name"], fathersName = fields["father"], dob = fields["dob"]?.let { parseDate(it, "PAN") }, panNumber = panNumber, imagePath = path, uploadedBy = user, uploadedAt = timestamp))
            }
            DocumentType.VOTER_ID -> {
                val voterIdNumber = fields["voter_id"] ?: throw IllegalArgumentException("Voter ID number is required and was not extracted.")
                voterIdDataRepository.save(VoterIdData(name = fields["name"], gender = fields["gender"], dob = fields["date"]?.let { parseDate(it, "Voter ID") }, voterIdNumber = voterIdNumber, imagePath = path, uploadedBy = user, uploadedAt = timestamp))
            }
            DocumentType.PASSPORT -> {
                val passportNumber = fields["passport_number"] ?: throw IllegalArgumentException("Passport number is required and was not extracted.")
                passportDataRepository.save(PassportData(name = fields["name"], gender = fields["gender"], dob = fields["dob"]?.let { parseDate(it, "Passport DOB") }, passportNumber = passportNumber, expiryDate = fields["expiry"]?.let { parseDate(it, "Passport Expiry Date") }, imagePath = path, uploadedBy = user, uploadedAt = timestamp))
            }
            DocumentType.UNKNOWN -> throw IllegalStateException("Should not attempt to save UNKNOWN document type.")
        }
    }

    fun getPanDataById(id: Long): PanData? = panDataRepository.findById(id).orElse(null)
    fun getVoterIdDataById(id: Long): VoterIdData? = voterIdDataRepository.findById(id).orElse(null)
    fun getPassportDataById(id: Long): PassportData? = passportDataRepository.findById(id).orElse(null)

    private fun storeFileToTemp(file: MultipartFile): Pair<File, String> {
        val uniqueFilename = "${System.currentTimeMillis()}_${UUID.randomUUID()}_${file.originalFilename}"
        val tempFilePath = tempFileLocation.resolve(uniqueFilename)
        file.inputStream.use { input -> Files.copy(input, tempFilePath, StandardCopyOption.REPLACE_EXISTING) }
        return Pair(tempFilePath.toFile(), uniqueFilename)
    }

    private fun moveFileToFinal(tempFile: File, docType: DocumentType, timestamp: LocalDateTime, uniqueFilename: String): Path {
        val dateDir = timestamp.format(timestampDirFormatter)
        val finalDir = fileStorageLocation.resolve(docType.name).resolve(dateDir)
        Files.createDirectories(finalDir)
        val finalPath = finalDir.resolve(uniqueFilename)
        return Files.move(tempFile.toPath(), finalPath, StandardCopyOption.REPLACE_EXISTING)
    }

    private fun convertPdfToImage(pdfFile: File): File {
        val pngFile = Files.createTempFile(tempFileLocation, "converted-", ".png").toFile()
        Loader.loadPDF(pdfFile).use { document ->
            val renderer = PDFRenderer(document)
            val image: BufferedImage = renderer.renderImageWithDPI(0, 300f)
            ImageIO.write(image, "PNG", pngFile)
        }
        return pngFile
    }

    private fun isPdf(multipartFile: MultipartFile, file: File): Boolean =
        "application/pdf".equals(multipartFile.contentType, ignoreCase = true) || "pdf".equals(file.extension, ignoreCase = true)

    private fun parseDate(dateStr: String, docTypeName: String): LocalDate {
        try {
            return LocalDate.parse(dateStr, dateFormatter)
        } catch (e: Exception) {
            throw IllegalArgumentException("Invalid date format '$dateStr' for $docTypeName. Expected dd/MM/yyyy.")
        }
    }

    private fun logDocument(username: String, fileName: String, docType: DocumentType, status: DocumentStatus, details: String? = null) {
        documentLogRepository.save(DocumentLog(
            uploader = username,
            fileName = Paths.get(fileName).fileName.toString(),
            docType = docType,
            status = status,
            createdAt = LocalDateTime.now()
        ))
    }

    private fun mapPythonDocTypeToEnum(pythonDocType: String): DocumentType =
        when (pythonDocType.lowercase()) {
            "pan" -> DocumentType.PAN_CARD
            "passport" -> DocumentType.PASSPORT
            "voterid_new", "voterid_old" -> DocumentType.VOTER_ID
            else -> DocumentType.UNKNOWN
        }

    // A private helper class to adapt a standard File to a MultipartFile for the Feign client.
    private class FileToMultipartFileAdapter(private val file: File) : MultipartFile {
        override fun getName(): String = "files"
        override fun getOriginalFilename(): String = file.name
        override fun getContentType(): String? = Files.probeContentType(file.toPath())
        override fun isEmpty(): Boolean = file.length() == 0L
        override fun getSize(): Long = file.length()
        override fun getBytes(): ByteArray = file.readBytes()
        override fun getInputStream(): InputStream = file.inputStream()
        override fun transferTo(dest: File) {
            file.copyTo(dest, overwrite = true)
        }
    }
}