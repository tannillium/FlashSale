package com.flashsale.dto;
import jakarta.validation.constraints.NotBlank;
public record PurchaseRequest(@NotBlank String productId,@NotBlank String customerId){}
