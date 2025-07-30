package com.example.docprocessor.service.client

import com.example.docprocessor.dto.PythonOcrResponse
import org.springframework.cloud.openfeign.FeignClient
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.multipart.MultipartFile

// A declarative REST client for the Python OCR microservice, powered by Spring Cloud OpenFeign.
// Spring automatically generates a proxy class that implements this interface,
// handling all the boilerplate HTTP communication code for us.
@FeignClient(name = "ocr-service", url = "\${ocr.service.url}")
interface OcrServiceClient {

    @PostMapping(value = ["/ocr/process/"], consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun processDocument(@RequestPart("files") file: MultipartFile): PythonOcrResponse
}