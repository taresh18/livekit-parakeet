# LiveKit NVIDIA NeMo STT Plugin

A high-performance speech-to-text plugin for LiveKit agents using NVIDIA NeMo Parakeet and Canary models for accurate and fast speech recognition.

## Features

- **Dual Model Support**: Compatible with both NVIDIA NeMo Parakeet and Canary STT models
- **Ultra-Low Latency**: Sub-100ms latency with Parakeet on RTX 4090
- **High Accuracy**: Enterprise-grade speech recognition using NVIDIA's state-of-the-art models
- **Optimized Performance**: Connection pooling and keep-alive optimizations for production use
- **LiveKit Integration**: Seamless integration with LiveKit agents framework

## Requirements

- LiveKit Agents v1.2 or higher
- NVIDIA [NeMo STT server](https://github.com/taresh18/parakeet-FastAPI) instance 

## Performance Benchmarks

| Model | Hardware | Latency | Use Case |
|-------|----------|---------|----------|
| **Parakeet** | RTX 4090 | <100ms | Real-time applications |
| **Canary** | RTX 4090 | <150ms | High-accuracy transcription |

## Installation

1. Clone or download this plugin into your LiveKit-based agents project root directory
2. Set up the NVIDIA NeMo STT server using the FastAPI implementation
3. Ensure your server is running and accessible

## Server Setup

Use the FastAPI server implementation for both Parakeet and Canary models:

**Repository**: [taresh18/parakeet-FastAPI](https://github.com/taresh18/parakeet-FastAPI)

This server provides unified support for both NVIDIA NeMo models with optimized inference endpoints.

## Usage

Initialize your agent session with the CanarySTT plugin:

```python
from parakeet import CanarySTT

session = AgentSession(
    # ... other configuration
    stt=CanarySTT(
        server_url="<your_server_url>",  # e.g., "http://localhost:8989"
    )
)
```
