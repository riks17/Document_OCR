package com.example.docprocessor.model

import jakarta.persistence.*

// Represents the role of a user in the system.
enum class Role {
    USER,
    ADMIN,
    SUPERADMIN
}

@Entity
@Table(name = "users")
data class User(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Long? = null,

    @Column(unique = true, nullable = false)
    var username: String,

    @Column(nullable = false)
    var password: String,

    @Enumerated(EnumType.STRING) // This will now store "USER", "ADMIN", etc.
    @Column(nullable = false)
    var role: Role
)