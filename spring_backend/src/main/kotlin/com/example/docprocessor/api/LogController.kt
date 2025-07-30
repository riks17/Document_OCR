package com.example.docprocessor.api

import com.example.docprocessor.model.DocumentLog
import com.example.docprocessor.repository.DocumentLogRepository
import org.springframework.data.domain.Sort
import org.springframework.http.ResponseEntity
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.CrossOrigin
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@CrossOrigin(origins = ["*"], maxAge = 3600)
@RestController
@RequestMapping("/api/logs")
class LogController(private val documentLogRepository: DocumentLogRepository) {

    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    fun getAllDocumentLogs(): ResponseEntity<List<DocumentLog>> {
        val logs = documentLogRepository.findAll(Sort.by(Sort.Direction.DESC, "createdAt"))
        return ResponseEntity.ok(logs)
    }
}