CREATE TABLE sale_orders (id UUID PRIMARY KEY, product_id VARCHAR(128) NOT NULL, customer_id VARCHAR(128) NOT NULL, idempotency_key VARCHAR(255) NOT NULL, status VARCHAR(32) NOT NULL, created_at TIMESTAMPTZ NOT NULL, CONSTRAINT uk_sale_orders_idempotency_key UNIQUE (idempotency_key));
CREATE INDEX ix_sale_orders_product_created ON sale_orders(product_id,created_at);
