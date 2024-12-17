mod gpu_monitor;
mod logger;

use anyhow::Result;
use candle_core::Device;

pub use gpu_monitor::{GpuMonitor, GpuMetrics};
pub use logger::{Logger, LogAction};

pub async fn init_gpu_monitor() -> Result<GpuMonitor<Logger>> {
    let device = Device::cuda_if_available(0)
        .expect("CUDA GPU is required but not available");
    let logger = Logger::new("kafka:29092", "gpu-monitor")?;
    let monitor = GpuMonitor::new(device, logger);
    
    // Verify GPU computation on startup
    monitor.verify_gpu_compute().await?;
    
    Ok(monitor)
}

// Note: GPU functionality is tested in gpu_monitor.rs using mocks
// Integration tests with real Kafka are handled via docker-compose
 