package com.example.docprocessor.dto

import com.example.docprocessor.model.Role
import jakarta.validation.constraints.NotBlank

// DTO for the user login request.
data class LoginRequest(@field:NotBlank val username: String, @field:NotBlank val password: String)

// DTO for the public user registration endpoint.
// It intentionally lacks a 'role' field to prevent privilege escalation.
data class SignupRequest(@field:NotBlank val username: String, @field:NotBlank val password: String)

// DTO for the secure, admin-only endpoint to create users with specific roles.
data class PrivilegedUserCreateRequest(
    @field:NotBlank val username: String,
    @field:NotBlank val password: String,
    val role: Role
)

// DTO for the response after a successful login, containing the JWT and user info.
data class JwtResponse(
    val token: String,
    val id: Long,
    val username: String,
    val roles: List<String>
)

// A generic DTO for sending simple, single-message responses.
data class MessageResponse(val message: String)