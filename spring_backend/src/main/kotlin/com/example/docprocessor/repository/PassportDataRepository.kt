package com.example.docprocessor.repository

import com.example.docprocessor.model.PassportData
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.Optional

// Provides data access methods for PassportData entities.
@Repository
interface PassportDataRepository : JpaRepository<PassportData, Long> {
    fun findByPassportNumber(passportNumber: String): Optional<PassportData>
}