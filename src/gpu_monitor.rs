use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;
use std::process::Command;
use crate::logger::LogAction;

pub struct GpuMetrics {
    pub execution_time_ms: u64,
    pub device_type: String,
    pub operation_name: String,
}

pub struct GpuMonitor<L: LogAction> {
    device: Device,
    logger: L,
}

impl<L: LogAction> GpuMonitor<L> {
    pub fn new(device: Device, logger: L) -> Self {
        Self { device, logger }
    }

    pub fn verify_gpu_info(&self) -> Result<String> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,gpu_uuid,pci.bus_id,clocks.max.sm,clocks.max.mem",
                "--format=csv,noheader"
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute nvidia-smi: {}", e))?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("nvidia-smi failed: {}", error));
        }

        let info = String::from_utf8(output.stdout)
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in nvidia-smi output: {}", e))?;
        
        Ok(info.trim().to_string())
    }

    pub async fn log_operation<T, F>(&self, operation_name: &str, input_shape: &[i64], operation: F) -> Result<T> 
    where
        F: FnOnce() -> Result<T>,
    {
        let start_time = Instant::now();
        let result = operation()?;
        let execution_time = start_time.elapsed();

        let metrics = GpuMetrics {
            execution_time_ms: execution_time.as_millis() as u64,
            device_type: format!("{:?}", self.device),
            operation_name: operation_name.to_string(),
        };

        self.logger.log_action(
            "gpu_operation",
            "gpu_monitor",
            &format!(
                "Operation: {}, Input Shape: {:?}, Time: {}ms, Device: {}",
                metrics.operation_name,
                input_shape,
                metrics.execution_time_ms,
                metrics.device_type
            ),
        ).await?;

        Ok(result)
    }

    pub async fn verify_gpu_compute(&self) -> Result<bool> {     
        // Create large tensors to ensure GPU utilization
        let a = Tensor::randn(0f32, 1.0f32, (2000, 2000), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create tensor A: {}", e))?;
        
        let b = Tensor::randn(0f32, 1.0f32, (2000, 2000), &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create tensor B: {}", e))?;

        // Test on GPU first
        println!("Starting GPU matrix multiply...");
        let gpu_start = Instant::now();
        let gpu_result = a.matmul(&b)
            .map_err(|e| anyhow::anyhow!("GPU matmul failed: {}", e))?;
        let gpu_time = gpu_start.elapsed();
        println!("GPU operation took: {:?}", gpu_time);

        // Log the operation after timing
        self.log_operation(
            "matrix_multiply",
            &[2000, 2000],
            || Ok(gpu_result.clone())
        ).await?;

        println!("Starting CPU comparison...");
        // Test on CPU for comparison
        let device_cpu = Device::Cpu;
        let a_cpu = Tensor::randn(0f32, 1.0f32, (2000, 2000), &device_cpu)
            .map_err(|e| anyhow::anyhow!("Failed to create CPU tensor A: {}", e))?;
        let b_cpu = Tensor::randn(0f32, 1.0f32, (2000, 2000), &device_cpu)
            .map_err(|e| anyhow::anyhow!("Failed to create CPU tensor B: {}", e))?;
        
        let cpu_start = Instant::now();
        let _cpu_result = a_cpu.matmul(&b_cpu)
            .map_err(|e| anyhow::anyhow!("CPU matmul failed: {}", e))?;
        let cpu_time = cpu_start.elapsed();
        println!("CPU operation took: {:?}", cpu_time);

        // Verify GPU result is valid
        let sum = gpu_result.sum_all()
            .map_err(|e| anyhow::anyhow!("Failed to sum GPU result: {}", e))?;
        
        let scalar = sum.to_scalar::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert sum to scalar: {}", e))?;
        
        println!("Comparing times - GPU: {:?}, CPU: {:?}", gpu_time, cpu_time);
        Ok(!scalar.is_nan() && gpu_time * 2 < cpu_time)
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    // Simple mock logger
    #[derive(Clone)]
    pub struct MockLogger {}

    impl MockLogger {
        pub fn new() -> Self {
            Self {}
        }
    }

    #[async_trait::async_trait]
    impl LogAction for MockLogger {
        async fn log_action(&self, _action: &str, _component: &str, _details: &str) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::mock::MockLogger;

    fn get_gpu() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(device) => {
                println!("Successfully initialized CUDA device: {:?}", device);
                Some(device)
            },
            Err(e) => {
                println!("Failed to initialize CUDA device: {}", e);
                None
            }
        }
    }

    #[tokio::test]
    async fn test_gpu_detection() {
        println!("Starting GPU detection test");
        
        println!("Getting GPU device...");
        let device = get_gpu().expect("This test requires a GPU");
        println!("Got device: {:?}", device);
        
        println!("Creating mock logger...");
        let logger = MockLogger::new();
        
        println!("Creating GPU monitor...");
        let monitor = GpuMonitor::new(device, logger);
        
        println!("Starting GPU verification...");
        let result = monitor.verify_gpu_compute().await;
        
        match result {
            Ok(is_working) => {
                println!("GPU verification completed with result: {}", is_working);
                assert!(is_working, "GPU performance verification returned false");
            },
            Err(e) => {
                println!("GPU verification failed with error: {:#?}", e);
                panic!("GPU verification error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tensor_operations() {
        let device = get_gpu().expect("This test requires a GPU");
        println!("Using device: {:?}", device);
        let logger = MockLogger::new();
        let monitor = GpuMonitor::new(device.clone(), logger);

        println!("Creating test tensors...");
        let a = Tensor::randn(0f32, 1.0f32, (100, 100), &device)
            .expect("Failed to create tensor A");
        let b = Tensor::randn(0f32, 1.0f32, (100, 100), &device)
            .expect("Failed to create tensor B");

        println!("Testing GPU operation...");
        let result = monitor.log_operation(
            "matrix_multiply",
            &[100i64, 100i64],
            || a.matmul(&b).map_err(|e| anyhow::anyhow!("GPU matmul failed: {}", e))
        ).await.expect("GPU operation failed");

        assert_eq!(result.dims(), &[100, 100], "Unexpected result dimensions");
        
        println!("Testing CPU operation...");
        let cpu_start = Instant::now();
        let a_cpu = Tensor::randn(0f32, 1.0f32, (100, 100), &Device::Cpu)
            .expect("Failed to create CPU tensor A");
        let b_cpu = Tensor::randn(0f32, 1.0f32, (100, 100), &Device::Cpu)
            .expect("Failed to create CPU tensor B");
        let _cpu_result = a_cpu.matmul(&b_cpu)
            .expect("CPU matmul failed");
        let cpu_time = cpu_start.elapsed();
        println!("CPU operation took: {:?}", cpu_time);

        println!("Testing GPU timing...");
        let gpu_start = Instant::now();
        let _ = a.matmul(&b).expect("GPU timing test failed");
        let gpu_time = gpu_start.elapsed();
        println!("GPU operation took: {:?}", gpu_time);
        
        assert!(
            gpu_time < cpu_time,
            "GPU ({:?}) was not faster than CPU ({:?})",
            gpu_time,
            cpu_time
        );
    }

    #[test]
    fn test_gpu_info() {
        let device = get_gpu().expect("This test requires a GPU");
        let logger = MockLogger::new();
        let monitor = GpuMonitor::new(device, logger);
        
        let gpu_info = monitor.verify_gpu_info().expect("Failed to get GPU info");
        println!("GPU Info: {}", gpu_info);
        assert!(!gpu_info.is_empty(), "GPU info should not be empty");
    }
} 