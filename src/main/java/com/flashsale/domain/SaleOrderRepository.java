package com.flashsale.domain;
import java.util.UUID; import org.springframework.data.jpa.repository.JpaRepository;
public interface SaleOrderRepository extends JpaRepository<SaleOrder,UUID> { boolean existsByIdempotencyKey(String idempotencyKey); }
