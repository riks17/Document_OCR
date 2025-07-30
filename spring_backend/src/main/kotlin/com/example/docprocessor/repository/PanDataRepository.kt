package com.example.docprocessor.repository

import com.example.docprocessor.model.PanData
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.Optional

// Provides data access methods for PanData entities.
@Repository
interface PanDataRepository : JpaRepository<PanData, Long> {
    fun findByPanNumber(panNumber: String): Optional<PanData>
}