package com.flashsale.messaging;
import java.util.UUID;
public record OrderEvent(UUID orderId,String productId,String customerId,String idempotencyKey){}
