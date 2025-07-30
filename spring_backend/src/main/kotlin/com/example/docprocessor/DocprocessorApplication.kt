package com.example.docprocessor

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.cloud.openfeign.EnableFeignClients // <-- IMPORT THIS

@SpringBootApplication
@EnableFeignClients
class DocprocessorApplication

fun main(args: Array<String>) {
	runApplication<DocprocessorApplication>(*args)
}