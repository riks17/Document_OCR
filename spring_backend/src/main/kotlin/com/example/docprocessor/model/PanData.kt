package com.example.docprocessor.model

import jakarta.persistence.*
import java.time.LocalDate
import java.time.LocalDateTime

// Represents the structured data extracted from a PAN Card.
@Entity
@Table(name = "pan_data")
data class PanData(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Long? = null,
    var name: String?,
    var fathersName: String?,
    var dob: LocalDate?,
    @Column(unique = true, nullable = false, length = 10)
    var panNumber: String,
    @Column(nullable = false)
    var imagePath: String,
    @Column(nullable = false)
    var uploadedBy: String,
    @Column(nullable = false)
    var uploadedAt: LocalDateTime = LocalDateTime.now()
)