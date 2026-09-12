package com.flashsale.service;
import java.util.List; import org.springframework.data.redis.core.StringRedisTemplate; import org.springframework.data.redis.core.script.DefaultRedisScript; import org.springframework.stereotype.Service;
@Service public class InventoryService { private static final DefaultRedisScript<Long> RESERVE=new DefaultRedisScript<>("""
if redis.call('SET',KEYS[2],'1','NX','EX',ARGV[2]) == false then return -1 end
local current=tonumber(redis.call('GET',KEYS[1]) or '-1')
if current <= 0 then redis.call('DEL',KEYS[2]); return 0 end
redis.call('DECR',KEYS[1]); return 1
""",Long.class); private final StringRedisTemplate redis; public InventoryService(StringRedisTemplate redis){this.redis=redis;} public long reserve(String productId,String idempotencyKey){Long result=redis.execute(RESERVE,List.of(stockKey(productId),idemKey(idempotencyKey)),"1","86400");return result==null?0:result;} public long available(String productId){String raw=redis.opsForValue().get(stockKey(productId));return raw==null?0:Long.parseLong(raw);} public void seed(String productId,long quantity){redis.opsForValue().set(stockKey(productId),Long.toString(quantity));} private String stockKey(String productId){return "flashsale:stock:"+productId;} private String idemKey(String key){return "flashsale:idem:"+key;} }
