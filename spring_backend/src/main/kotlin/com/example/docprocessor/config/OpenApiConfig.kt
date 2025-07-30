package com.example.docprocessor.config

import io.swagger.v3.oas.models.Components
import io.swagger.v3.oas.models.OpenAPI
import io.swagger.v3.oas.models.info.Info
import io.swagger.v3.oas.models.security.SecurityRequirement
import io.swagger.v3.oas.models.security.SecurityScheme
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class OpenApiConfig {

    @Bean
    fun customOpenAPI(): OpenAPI {
        val securitySchemeName = "bearerAuth"
        return OpenAPI()
            .info(
                Info()
                    .title("DocProcessor API")
                    .version("1.0.0")
                    .description(
                        """
                        API for processing and managing identity documents using OCR.
                        
                        This API is secured with JWT Bearer tokens. To use the secured endpoints below, 
                        first log in via the `/api/auth/login` endpoint to receive a token. 
                        Then, click the 'Authorize' button on this page and enter your token in the format: `Bearer {your_token}`.
                        """.trimIndent()
                    )
            )
            // Add the 'Authorize' button to the UI
            .addSecurityItem(SecurityRequirement().addList(securitySchemeName))
            // Define the security scheme (JWT Bearer token)
            .components(
                Components()
                    .addSecuritySchemes(
                        securitySchemeName,
                        SecurityScheme()
                            .name(securitySchemeName)
                            .type(SecurityScheme.Type.HTTP)
                            .scheme("bearer")
                            .bearerFormat("JWT")
                    )
            )
    }
}