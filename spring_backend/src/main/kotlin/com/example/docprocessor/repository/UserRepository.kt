package com.example.docprocessor.repository

import com.example.docprocessor.model.Role
import com.example.docprocessor.model.User
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.Optional

// Provides data access methods for User entities.
@Repository
interface UserRepository : JpaRepository<User, Long> {
    fun findByUsername(username: String): Optional<User>
    fun existsByUsername(username: String): Boolean
    fun findFirstByRole(role: Role): Optional<User>
}