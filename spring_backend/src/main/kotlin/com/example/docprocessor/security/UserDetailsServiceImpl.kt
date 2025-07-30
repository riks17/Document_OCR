package com.example.docprocessor.security

import com.example.docprocessor.repository.UserRepository
import org.springframework.security.core.userdetails.UserDetails
import org.springframework.security.core.userdetails.UserDetailsService
import org.springframework.security.core.userdetails.UsernameNotFoundException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

// This service is the bridge between Spring Security and our user data model.
@Service
class UserDetailsServiceImpl(private val userRepository: UserRepository) : UserDetailsService {

    // This method is called by Spring Security during the authentication process.
    // It is responsible for fetching a user from the database by their username.
    @Transactional(readOnly = true)
    override fun loadUserByUsername(username: String): UserDetails {
        val user = userRepository.findByUsername(username)
            .orElseThrow { UsernameNotFoundException("User Not Found with username: $username") }

        // We wrap our own User entity in our custom UserDetailsImpl class.
        return UserDetailsImpl(user)
    }
}