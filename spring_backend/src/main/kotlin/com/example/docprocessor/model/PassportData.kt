package com.example.docprocessor.model

import jakarta.persistence.*
import java.time.LocalDate
import java.time.LocalDateTime

// Represents the structured data extracted from a Passport.
@Entity
@Table(name = "passport_data")
data class PassportData(
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Long? = null,
    var name: String?,
    @Column(length = 10)
    var gender: String?,
    var dob: LocalDate?,
    @Column(unique = true, nullable = false, length = 20)
    var passportNumber: String,
    var expiryDate: LocalDate?,
    @Column(nullable = false)
    var imagePath: String,
    @Column(nullable = false)
    var uploadedBy: String,
    @Column(nullable = false)
    var uploadedAt: LocalDateTime = LocalDateTime.now()
)