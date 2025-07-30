package com.example.docprocessor.config

import com.example.docprocessor.model.Role
import com.example.docprocessor.model.User
import com.example.docprocessor.repository.UserRepository
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.CommandLineRunner
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Component

// A bootstrap component that runs on application startup.
// Its purpose is to ensure that at least one SUPERADMIN user exists,
// preventing the system from being locked out.
@Component
class SuperAdminInitializer(
    private val userRepository: UserRepository,
    private val passwordEncoder: PasswordEncoder,
    @Value("\${app.superadmin.username}") private val superAdminUsername: String,
    @Value("\${app.superadmin.password}") private val superAdminPassword: String
) : CommandLineRunner {

    private val logger = LoggerFactory.getLogger(SuperAdminInitializer::class.java)

    override fun run(vararg args: String?) {
        if (userRepository.findFirstByRole(Role.SUPERADMIN).isEmpty) {
            logger.info("No SuperAdmin account found. Creating a default SuperAdmin user.")

            val superAdmin = User(
                username = superAdminUsername,
                password = passwordEncoder.encode(superAdminPassword),
                role = Role.SUPERADMIN
            )
            userRepository.save(superAdmin)

            logger.warn("=================================================================")
            logger.warn("  >>> Default SuperAdmin Account Created <<<")
            logger.warn("  Username: $superAdminUsername")
            logger.warn("  Password: $superAdminPassword")
            logger.warn("  This is for development purposes. Please use a secure password in production.")
            logger.warn("=================================================================")
        } else {
            logger.info("SuperAdmin account already exists. Skipping creation.")
        }
    }
}