# Architecture notes

```text
client → POST /orders → Redis Lua reserve → Kafka order event → worker → PostgreSQL
```

Redis runs one atomic Lua script to claim the idempotency key and decrement stock only if stock is positive. Kafka is partitioned by product ID, allowing parallel processing across products. A PostgreSQL uniqueness constraint on `idempotency_key` protects against Kafka redelivery.

An order response is asynchronous: `202 ACCEPTED` means it has been reserved and queued for processing. In production, configure producer acknowledgements and a transactional outbox/reconciliation worker to recover from a Kafka outage after a Redis reservation.

The API needs `Idempotency-Key` on `POST /api/v1/orders`. `PUT /api/v1/inventory/{productId}?quantity=N` is a demo/admin stock seed endpoint and should be protected or removed in production.
