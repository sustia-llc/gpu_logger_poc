FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    cmake \
    git \
    nvidia-utils-560 \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create workspace
WORKDIR /app

# Set environment variables
ENV CUDA_COMPUTE_CAP=86
ENV RUST_BACKTRACE=1
ENV RUST_LOG=debug

# Copy Rust project files
COPY . .

# Build release binary
RUN cargo build --release

# Runtime stage
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=builder /app/target/release/llm-service /usr/local/bin/

WORKDIR /app

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=all
ENV RUST_BACKTRACE=1
ENV RUST_LOG=debug

CMD ["llm-service"] 