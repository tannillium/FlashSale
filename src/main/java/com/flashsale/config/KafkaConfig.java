package com.flashsale.config;
import org.apache.kafka.clients.admin.NewTopic; import org.springframework.context.annotation.*; import org.springframework.kafka.config.TopicBuilder;
@Configuration public class KafkaConfig { public static final String ORDER_TOPIC="flashsale.orders.v1"; @Bean NewTopic orderTopic(){return TopicBuilder.name(ORDER_TOPIC).partitions(6).replicas(1).build();} }
