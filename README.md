# FlashSale — Distributed Flash-Sale Order System

## The Elevator Pitch

Imagine it’s Black Friday, or tickets just dropped for a massive concert. You have 1,000 items in stock, but 100,000 people are mashing the “Buy” button at the exact same millisecond. If you just connect a standard API to a database, the system will crash, or worse—you’ll accidentally sell 5,000 items when you only have 1,000.

I built FlashSale to solve this exact problem. It’s a high-concurrency, event-driven ordering system that guarantees we never oversell a single item, while keeping the user experience lightning fast.

## Tech Stack

- Core: Java, Spring Boot
- Data & Messaging: PostgreSQL, Redis, Kafka
- Testing & Ops: Docker, JUnit, Testcontainers, JPA

## How It Works (The Flow)

Instead of making the user wait for a slow database to process everything, I split the process into two parts: fast validation and background processing.

1. The Click: A user hits the Spring Boot API to buy a product.
2. The Fast Check (Redis): Going to a standard database is too slow for 100k users. Instead, the API checks Redis (super-fast memory). Redis uses an atomic counter to instantly check inventory. If the stock is 0, the user gets a “Sold Out” message immediately.
3. The Queue (Kafka): If Redis says there is stock, we deduct one from Redis and instantly drop an “Order Event” into Kafka. We then immediately tell the user, “Success! Your order is being processed.”
4. The Background Workers: Meanwhile, backend Order Workers are listening to Kafka. They pick up the orders at a safe, steady pace and write the final details to PostgreSQL.

Think of Kafka like a ticket wheel at a busy diner. The waiter (Spring Boot) takes your order instantly and puts the ticket on the wheel (Kafka), so they can quickly help the next customer. The chef (Workers) pulls tickets off the wheel and cooks (writes to Postgres) at their own safe pace without getting overwhelmed.

## Key Technical Challenges & How I Solved Them

### 1. Preventing “Overselling” (Race Conditions & Concurrency)

When thousands of requests hit at once, traditional databases can get confused and sell the same item to two different people. By keeping the live inventory count in Redis and using atomic operations (or distributed locks), I ensured that stock only ever goes down by exactly one per valid request.

### 2. Handling the Traffic Spike (Event-Driven Architecture)

If 100,000 people try to write to PostgreSQL simultaneously, the database will lock up and crash. Kafka acts as a giant shock absorber. It absorbs the massive spike in traffic and lets the database process the orders sequentially in the background.

### 3. The “Panic Click” Problem (Idempotency)

What happens if a user gets impatient and double-clicks the “Buy” button? I implemented idempotency keys (like a unique request ID). If the system sees the same ID twice, it simply ignores the second request, ensuring a user isn’t accidentally charged twice.

### 4. Making Sure It Actually Works (Integration Testing)

Mocking is great, but to prove this works under pressure, I used Testcontainers. During automated testing (JUnit), Docker spins up real instances of Redis, Kafka, and PostgreSQL, runs simulated high-traffic orders through the system, and verifies that the final database count is mathematically perfect.

## Why I Built This (What I Learned)

Building a standard CRUD app is easy, but making an app survive a massive traffic spike requires a totally different mindset. This project taught me how real-world distributed systems work. I learned how to manage race conditions, why message brokers like Kafka are essential for scaling, and how to protect a database from getting crushed under load.

---

## Project Summary

FlashSale is a distributed flash-sale system designed to handle extreme concurrency without losing consistency. The idea is simple: validate inventory quickly, push events asynchronously, and let background workers persist the final order state safely. This architecture is a strong example of how to scale high-throughput e-commerce flows while maintaining correctness.
