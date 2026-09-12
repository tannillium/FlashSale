package com.flashsale.dto;
import java.util.UUID;
public record PurchaseResponse(UUID orderId,String status,String message){}
