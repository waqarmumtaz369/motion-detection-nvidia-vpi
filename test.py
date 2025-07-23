import pika
import json
 
# Sample event object
event_data = {
    "camera_id": 1,
    "event": "motion_detected",
    "timestamp": "2025-07-18T12:34:56Z",
    "confidence": 0.87,
}
 
# Connect to RabbitMQ
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="172.26.100.163",
        port=5672,
        virtual_host="/",
        credentials=pika.PlainCredentials("admin", "admin"),
    )
)
 
channel = connection.channel()
channel.queue_declare(queue="motion", durable=True)
 
# Publish the message
channel.basic_publish(
    exchange="",
    routing_key="motion",
    body=json.dumps(event_data),
    properties=pika.BasicProperties(delivery_mode=2),
)
 
print("Published message:", event_data)
 
connection.close()