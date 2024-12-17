use anyhow::Result;
use gpu_logger_poc::init_gpu_monitor;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting LLM service...");
    
    // Initialize GPU monitor
    let monitor = init_gpu_monitor().await?;
    
    println!("Service initialized, entering main loop...");
    
    // Keep the service running
    loop {
        // Verify GPU computation every minute
        monitor.verify_gpu_compute().await?;
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
} 