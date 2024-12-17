# GPU Logger POC

A system for verifying GPU execution of ML models with immutable logging via Kafka.

## System Overview

### Core Components

1. **GPU Monitoring**
   - Requires CUDA-capable GPU
   - Hardware verification via nvidia-smi
   - Verifies GPU availability and computation
   - Measures operation execution times
   - Compares GPU vs CPU performance
   - Validates tensor operations on GPU

2. **Immutable Logging**
   - Uses Kafka for reliable logging
   - Timestamps all operations
   - Records operation details and performance metrics
   - Preserves operation history in kafka-data volume

3. **Performance Verification**
   - Hardware capabilities inspection via nvidia-smi
   - Operation execution times
   - Device type verification
   - GPU vs CPU performance comparison
   - Matrix multiplication benchmarking
     - 2000x2000 matrices for production verification
     - 100x100 matrices for faster unit tests

### Design Philosophy

The system prioritizes reliable GPU execution verification:
- Direct GPU monitoring through Candle
- Hardware verification through nvidia-smi
- Native Kafka integration for immutable logs
- Strict GPU requirements for tests
- Focus on GPU performance validation

## Setup Requirements

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# CUDA toolkit and nvidia-smi are required
# Install CUDA toolkit appropriate for your system
# See: https://developer.nvidia.com/cuda-downloads

# For Docker with GPU support
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker

# Kafka is required for logging (if running locally)
# Or use provided docker-compose setup
```

## Running Tests

```bash
# Run unit tests locally (requires CUDA GPU)
cargo test

# Run full integration tests with Kafka in Docker
docker-compose up test

# Tests will fail if no GPU is available
# All tests explicitly verify GPU performance
```

## Docker Environment

The project includes a complete Docker setup with:
- Kafka and Zookeeper services
- NVIDIA GPU runtime support
- Test container with cargo caching
- Health checks for dependencies
- Persistent Kafka logs in kafka-data volume

To use Docker for testing:
```bash
# Run integration tests
docker-compose up test

# Or run specific services
docker-compose up -d kafka  # Start Kafka only
docker-compose up llm-service  # Run main service
```

## Usage Example

```rust
use candle_core::Device;

let device = Device::cuda_if_available(0).expect("GPU required");
let monitor = GpuMonitor::new(device, logger);

// Verify GPU hardware
let gpu_info = monitor.verify_gpu_info()?;
println!("GPU Info: {}", gpu_info);  // Shows GPU name, UUID, clocks, etc.

// Verify GPU computation
let is_working = monitor.verify_gpu_compute().await?;
assert!(is_working, "GPU must perform significantly better than CPU");

// Log custom GPU operations
let result = monitor.log_operation(
    "matrix_multiply",
    &[2000, 2000],
    || perform_gpu_operation()
).await?;
```

## Monitoring Output

The system logs operations to Kafka with timestamps and unique keys:

```json
{
    "timestamp": "2024-01-XX...",
    "action": "gpu_operation",
    "component": "gpu_monitor",
    "details": "Operation: matrix_multiply, Input Shape: [2000, 2000], Time: 123ms, Device: Cuda(0)"
}
```

## Verification Methods

The system verifies GPU execution through:
1. Hardware verification using nvidia-smi
   - GPU name and UUID
   - PCI bus ID
   - Maximum SM and memory clocks
2. Performance verification
   - Explicit CUDA device requirement
   - Performance comparison using matrix multiplication
   - Validation of tensor operations
   - Requiring GPU performance to be at least 2x faster than CPU
   - No CPU fallback - tests fail if GPU is not available

## Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit and nvidia-smi installed
- Docker with NVIDIA Container Toolkit (for Docker usage)
- Kafka instance for logging (provided via Docker)
- Rust toolchain with CUDA support

## Possible next steps

- Model loading and inference support
- Model execution verification
- Performance metrics for model operations