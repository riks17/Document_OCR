package com.example.docprocessor.repository

import com.example.docprocessor.model.DocumentLog
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository

// Provides data access methods for DocumentLog entities.
@Repository
interface DocumentLogRepository : JpaRepository<DocumentLog, Long>