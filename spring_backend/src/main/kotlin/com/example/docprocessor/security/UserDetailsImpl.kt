package com.example.docprocessor.security

import com.example.docprocessor.model.User
import com.fasterxml.jackson.annotation.JsonIgnore
import org.springframework.security.core.GrantedAuthority
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.userdetails.UserDetails

// A custom implementation of UserDetails that wraps our own User entity.
// This acts as a bridge between our application's user model and Spring Security's framework.
class UserDetailsImpl(private val user: User) : UserDetails {

    fun getId(): Long = user.id!!

    override fun getAuthorities(): Collection<GrantedAuthority> {
        // This is a critical step. Spring Security's `hasRole()` check automatically
        // looks for an authority with the "ROLE_" prefix. We add it here so that our
        // database can store clean role names (e.g., "ADMIN").
        val authority = SimpleGrantedAuthority("ROLE_${user.role.name}")
        return listOf(authority)
    }

    @JsonIgnore
    override fun getPassword(): String = user.password

    override fun getUsername(): String = user.username

    // For this application, accounts are always considered active and non-expired.
    override fun isAccountNonExpired(): Boolean = true
    override fun isAccountNonLocked(): Boolean = true
    override fun isCredentialsNonExpired(): Boolean = true
    override fun isEnabled(): Boolean = true
}