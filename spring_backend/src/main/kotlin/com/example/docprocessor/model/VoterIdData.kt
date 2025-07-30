package com.example.docprocessor.model

import jakarta.persistence.*
import java.time.LocalDate
import java.time.LocalDateTime

// Represents the structured data extracted from a Voter ID card.
@Entity
@Table(name = "voter_id_data")
data class VoterIdData(
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Long? = null,
    var name: String?,
    @Column(length = 10)
    var gender: String?,
    var dob: LocalDate?,
    @Column(unique = true, nullable = false, length = 20)
    var voterIdNumber: String,
    @Column(nullable = false)
    var imagePath: String,
    @Column(nullable = false)
    var uploadedBy: String,
    @Column(nullable = false)
    var uploadedAt: LocalDateTime = LocalDateTime.now()
)