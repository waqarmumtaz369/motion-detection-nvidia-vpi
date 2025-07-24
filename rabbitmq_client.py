import pika
import json
from typing import Dict, Any
import logging
import os
from dotenv import load_dotenv
from typing import cast

class RabbitMQClient:
    def __init__(self, host: str = None, port: int = None, username: str = None, password: str = None, queue: str = None):
        """
        Initialize RabbitMQ client with connection parameters from environment variables if not provided.
        """
        self.host = host or os.environ.get("RABBITMQ_HOST")
        self.port = port or int(os.environ.get("RABBITMQ_PORT", 5672))
        self.username = username or os.environ.get("RABBITMQ_USERNAME")
        self.password = password or os.environ.get("RABBITMQ_PASSWORD")
        self.queue = queue or os.environ.get("RABBITMQ_QUEUE")
        self.connection = None
        self.channel = None
        self.logger = logging.getLogger(__name__)

        # Validate required config
        missing = [k for k, v in {"RABBITMQ_HOST": self.host, "RABBITMQ_USERNAME": self.username, "RABBITMQ_PASSWORD": self.password, "RABBITMQ_QUEUE": self.queue}.items() if not v]
        if missing:
            raise ValueError(f"Missing required RabbitMQ config: {', '.join(missing)}. Set these as environment variables or pass as arguments.")
        # Type assertions for type checker
        self.host = cast(str, self.host)
        self.username = cast(str, self.username)
        self.password = cast(str, self.password)
        self.queue = cast(str, self.queue)

    def connect(self) -> bool:
        """
        Establish connection to RabbitMQ server.
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue, durable=True)
            self.logger.info("Successfully connected to RabbitMQ")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            return False

    def publish_frame_details(self, frame_data: Dict[str, Any]) -> bool:
        """
        Publish frame processing details to RabbitMQ queue.
        Args:
            frame_data (Dict[str, Any]): Dictionary containing frame processing details
        Returns:
            bool: True if message was published successfully, False otherwise
        """
        try:
            if not self.connection or self.connection.is_closed:
                if not self.connect():
                    return False
            message = json.dumps(frame_data)
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                )
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish message: {str(e)}")
            return False

    def close(self):
        """
        Close the RabbitMQ connection.
        """
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            self.logger.info("RabbitMQ connection closed")

# Load environment variables from a .env file if present (for local development)
load_dotenv()
