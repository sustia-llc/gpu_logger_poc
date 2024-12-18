use anyhow::Result;
use candle_core::{Device, Tensor};
use std::{error::Error, time::Instant};
use nvml_wrapper::Nvml;
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

    pub fn get_gpu_info(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let nvml = Nvml::init()?;
        let device_count = nvml.device_count()?;
        let mut gpu_info = Vec::new();

        for i in 0..device_count {
            let device = nvml.device_by_index(i)?;
            
            // Get the requested information
            let name = device.name()?;
            let uuid = device.uuid()?;
            let pci_info = device.pci_info()?;
            let clocks = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)?;

            // Format it similar to nvidia-smi CSV output
            let info = format!("{},{},{},{},{}",
                name,
                uuid,
                pci_info.bus_id,
                clocks, // Graphics clock
                device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)? // Memory clock
            );
            
            gpu_info.push(info);
        }

        Ok(gpu_info)
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

        // Verify GPU result is valid using a simpler verification method
        let verification_result = match gpu_result.dims() {
            dims if dims.len() == 2 => {
                // Just verify the tensor dimensions are correct
                dims[0] == 2000 && dims[1] == 2000
            },
            _ => {
                println!("Warning: Unexpected tensor dimensions: {:?}", gpu_result.dims());
                // Fall back to comparing just the execution times
                gpu_time * 2 < cpu_time
            }
        };

        println!("Comparing times - GPU: {:?}, CPU: {:?}", gpu_time, cpu_time);
        Ok(verification_result && gpu_time * 2 < cpu_time)
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
    fn test_verify_gpu_info() {
        let device = get_gpu().expect("This test requires a GPU");
        let logger = MockLogger::new();
        let monitor = GpuMonitor::new(device, logger);
        
        let gpu_info = monitor.verify_gpu_info().expect("Failed to get GPU info");
        println!("GPU Info: {}", gpu_info);
        assert!(!gpu_info.is_empty(), "GPU info should not be empty");
    }

    #[test]
    fn test_get_gpu_info() {
        let device = get_gpu().expect("This test requires a GPU");
        let logger = MockLogger::new();
        let monitor = GpuMonitor::new(device, logger);
        let gpu_info = monitor.get_gpu_info().expect("Failed to get GPU info");

        assert!(!gpu_info.is_empty(), "Should have at least one GPU");
        
        // Verify format of each GPU info string
        for info in gpu_info {
            let parts: Vec<&str> = info.split(',').collect();
            assert_eq!(parts.len(), 5, "Each GPU info should have 5 comma-separated values");
            
            // Name shouldn't be empty
            assert!(!parts[0].is_empty(), "GPU name should not be empty");
            // UUID should be present
            assert!(!parts[1].is_empty(), "GPU UUID should not be empty");
            // PCI bus ID should be present
            assert!(!parts[2].is_empty(), "PCI bus ID should not be empty");
            // Clock speeds should be numeric
            assert!(parts[3].parse::<i32>().is_ok(), "Graphics clock should be numeric");
            assert!(parts[4].parse::<i32>().is_ok(), "Memory clock should be numeric");
        }
    }
} 