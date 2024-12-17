use anyhow::Result;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::ClientConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use chrono::Utc;

#[derive(Serialize, Deserialize)]
struct LogEntry {
    timestamp: String,
    action: String,
    component: String,
    details: String,
}

#[async_trait::async_trait]
pub trait LogAction {
    async fn log_action(&self, action: &str, component: &str, details: &str) -> Result<()>;
}

pub struct Logger {
    pub(crate) producer: FutureProducer,
    pub(crate) topic: String,
}

impl Logger {
    pub fn new(brokers: &str, topic: &str) -> Result<Self> {
        let mut config = ClientConfig::new();
        config
            .set("bootstrap.servers", brokers)
            .set("message.timeout.ms", "60000")
            .set("socket.timeout.ms", "60000")
            .set("request.timeout.ms", "60000");

        let producer: FutureProducer = config.create()?;
        Ok(Self {
            producer,
            topic: topic.to_string(),
        })
    }
}

#[async_trait::async_trait]
impl LogAction for Logger {
    async fn log_action(&self, action: &str, component: &str, details: &str) -> Result<()> {
        let log_entry = LogEntry {
            timestamp: Utc::now().to_rfc3339(),
            action: action.to_string(),
            component: component.to_string(),
            details: details.to_string(),
        };

        let payload = serde_json::to_string(&log_entry)?;

        self.producer
            .send(
                FutureRecord::to(&self.topic)
                    .payload(&payload)
                    .key(&format!("{}-{}", component, Utc::now().timestamp())),
                Duration::from_secs(30),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send to Kafka: {:?}", e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};

    #[tokio::test]
    #[ignore] // Run only in integration tests with Kafka available
    async fn test_logger() {
        let kafka_addr = std::env::var("KAFKA_BROKERS")
            .unwrap_or_else(|_| "kafka:29092".to_string());
        let topic = "test-topic";

        // Create topic if it doesn't exist
        let admin: AdminClient<_> = ClientConfig::new()
            .set("bootstrap.servers", &kafka_addr)
            .create()
            .unwrap();

        let new_topic = NewTopic::new(topic, 1, TopicReplication::Fixed(1));
        admin.create_topics(&[new_topic], &AdminOptions::new())
            .await
            .unwrap_or_default(); // Ignore if topic already exists

        // Test logging
        let logger = Logger::new(&kafka_addr, topic).unwrap();
        logger.log_action("test", "test_component", "test details").await.unwrap();
    }
}