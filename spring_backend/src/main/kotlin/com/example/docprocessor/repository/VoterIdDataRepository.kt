package com.example.docprocessor.repository

import com.example.docprocessor.model.VoterIdData
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.Optional

// Provides data access methods for VoterIdData entities.
@Repository
interface VoterIdDataRepository : JpaRepository<VoterIdData, Long> {
    fun findByVoterIdNumber(voterIdNumber: String): Optional<VoterIdData>
}