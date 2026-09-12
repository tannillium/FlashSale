# FlashSale: Distributed High-Concurrency Order System
Imagine it’s Black Friday. 1,000 limited-edition sneakers just dropped. 100,000 people smash the "Buy" button at the exact same millisecond.
If this API is connected directly to a traditional database, one of two things will happen:
The Crush: The database gets flooded, locks up, and crashes.
The Nightmare (Overselling): The database gets confused by race conditions and sells 5,000 pairs when only 1,000 exist.
FlashSale is an event-driven, distributed ordering system built to survive this exact chaos. It guarantees zero database crashes, lightning-fast response times for users, and mathematically perfect inventory management—even under massive load.
-> The Tech Stack
Core: Java, Spring Boot
Data & Cache: PostgreSQL, Redis
Messaging (Event-Driven): Apache Kafka
Testing & Ops: JUnit, Docker, Testcontainers, JPA
-> High-Level Architecture
Instead of making users wait for a slow database, FlashSale splits the process into Fast Validation and Background Processing.
code
Text
[100k Users] 
    │
    ▼
[Spring Boot API] ──(Fast Check)──> [Redis] (Atomic Inventory Counter)
    │
 (If Stock > 0)
    │
    ▼
[Apache Kafka] (The Shock Absorber / Queue)
    │
    ▼
[Order Workers] (Background Processing)
    │
    ▼
[PostgreSQL] (Final Source of Truth)
-> How It Works (The Story)
To solve the "Black Friday Nightmare," I gave different technologies very specific jobs:
1. The Bouncer (Redis)
Instead of making users wait for a traditional database to check inventory, Redis stands at the front door. Redis is incredibly fast memory. When 100,000 requests hit, Redis uses atomic operations to instantly hand out exactly 1,000 "digital wristbands." The remaining 99,000 users instantly get a polite "Sold Out" message. No database locks, no overselling.
2. The Shock Absorber (Kafka)
Once a user gets a wristband, we don't send them to the database yet. Their order is dropped into Kafka, a high-speed message broker. The moment the order hits Kafka, the API tells the user: "Success! Your order is being processed." The user walks away happy in less than 200 milliseconds.
3. The Backroom (Order Workers & PostgreSQL)
Meanwhile, backend Order Workers are listening to Kafka. They are shielded from the chaos of the internet. They pull orders out of Kafka at a safe, steady pace and write the final details to PostgreSQL. Because Kafka absorbed the massive traffic spike, the Postgres database never breaks a sweat.
-> Core Technical Challenges Solved
Preventing Race Conditions: By storing live inventory in Redis and using atomic decrements/distributed locks, the system guarantees that two concurrent threads can never claim the same item.
Handling the "Panic Double-Click" (Idempotency): If a user excitedly double-clicks the "Buy" button, we don't want to charge them twice. Every request generates a unique idempotency key. If the system sees a duplicate key, it simply ignores the extra clicks.
Proving It Works (Testcontainers): Mocking isn't enough for a system like this. The test suite uses Testcontainers to spin up real Docker instances of Redis, Kafka, and PostgreSQL. The integration tests simulate thousands of concurrent users hitting the API simultaneously, verifying that exactly 1,000 items are sold every single time.
-> Getting Started (Running Locally)
(Note: Add your specific setup instructions here)
Prerequisites:
Docker & Docker Compose
Java 17+
Maven / Gradle
To run the project:
Clone the repo: git clone https://github.com/yourusername/FlashSale.git
Start the infrastructure (Redis, Kafka, Postgres): docker-compose up -d
Run the Spring Boot application: ./mvnw spring-boot:run
Run the high-concurrency integration tests: ./mvnw test
