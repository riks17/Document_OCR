package com.example.docprocessor.service

import com.example.docprocessor.dto.PrivilegedUserCreateRequest
import com.example.docprocessor.model.User
import com.example.docprocessor.repository.UserRepository
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service

// This service handles all business logic related to user management.
@Service
class UserService(
    private val userRepository: UserRepository,
    private val passwordEncoder: PasswordEncoder
) {
    // Creates a new user with a specified role.
    // This should only be called from a secure, admin-only controller endpoint.
    fun createUserWithRole(request: PrivilegedUserCreateRequest): User {
        if (userRepository.existsByUsername(request.username)) {
            throw IllegalArgumentException("Error: Username '${request.username}' is already taken!")
        }

        val user = User(
            username = request.username,
            password = passwordEncoder.encode(request.password),
            role = request.role
        )
        return userRepository.save(user)
    }
}