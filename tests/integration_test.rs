use gpu_logger_poc::init_gpu_monitor;
use anyhow::Result;

#[tokio::test]
#[ignore] // Only run with docker-compose
async fn test_gpu_monitor_with_kafka() -> Result<()> {
    let monitor = init_gpu_monitor().await?;
    assert!(monitor.verify_gpu_compute().await?);
    Ok(())
} 