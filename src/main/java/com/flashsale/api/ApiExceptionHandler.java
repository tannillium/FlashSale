package com.flashsale.api;
import java.util.Map; import org.springframework.http.HttpStatus; import org.springframework.web.bind.annotation.*;
@RestControllerAdvice public class ApiExceptionHandler { @ExceptionHandler(IllegalArgumentException.class) @ResponseStatus(HttpStatus.BAD_REQUEST) Map<String,String> invalid(IllegalArgumentException e){return Map.of("message",e.getMessage());} }
