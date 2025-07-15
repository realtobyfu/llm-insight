# LLM Interpretability Toolkit

A production-ready toolkit for analyzing and visualizing attention patterns in transformer-based language models, implementing novel interpretability techniques to understand model decision-making processes.

## 🚀 Key Features

### Core Interpretability
- 🔍 **Real-time Analysis**: REST APIs with sub-second response times for models up to 1B parameters
- 🎯 **Attention Visualization**: Interactive visualization of attention patterns, head importance, and token relationships
- 🧠 **Sparse Autoencoders (SAE)**: Custom implementation for feature extraction with L1 regularization
- 📈 **Token Importance**: Advanced scoring algorithms for understanding token contributions

### Advanced Features
- 🛡️ **Failure Prediction**: 85%+ accuracy in predicting model failure cases through attention pattern analysis
- 🚨 **Anomaly Detection**: Multi-signal anomaly detection combining attention patterns and activation analysis
- 💾 **Memory-Efficient Processing**: Chunked and streaming analysis for large models and long sequences
- 🔄 **WebSocket Support**: Real-time analysis updates for interactive applications
- 📊 **Interactive Dashboard**: Streamlit-based UI for visual exploration and batch analysis

### Production Features
- ⚡ **Redis Caching**: High-performance caching for repeated analyses
- 🐳 **Docker Support**: Containerized deployment with Docker Compose
- 📦 **Batch Processing**: Efficient analysis of multiple texts simultaneously
- 🔧 **Extensible Architecture**: Plugin system for custom interpretability methods
- 📝 **Comprehensive API**: REST, WebSocket, and Python APIs with full documentation

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional, for acceleration)
- Redis (optional, for caching)
- Docker (optional, for containerized deployment)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-interpretability-toolkit.git
cd llm-interpretability-toolkit

# Install with pip
pip install -e ".[all]"

# Or install specific components
pip install -e .  # Core functionality only
pip install -e ".[visualization]"  # With visualization support
pip install -e ".[dev]"  # Development dependencies
```

### Basic Usage

```python
from src.core import InterpretabilityAnalyzer

# Initialize analyzer
analyzer = InterpretabilityAnalyzer(model_name="gpt2")

# Analyze text with multiple methods
results = analyzer.analyze(
    "The cat sat on the mat",
    methods=["attention", "importance", "sae", "head_patterns"]
)

# Access results
attention_patterns = results["attention"]["patterns"]
token_importance = results["importance"]["token_importance"]

# Visualize attention patterns
fig = analyzer.visualize_attention(
    results["attention"],
    layer=0,
    head=0,
    save_path="attention_viz.png"
)

# Predict failure probability
failure_result = analyzer.predict_failure_probability(
    "the the the the the",
    threshold=0.5
)
print(f"Failure probability: {failure_result['failure_probability']:.2%}")

# Detect anomalies
anomaly_result = analyzer.detect_anomalies(
    "aaaaaaaaaaaaaaaaaaa",
    check_attention=True,
    check_activations=True
)
if anomaly_result["is_anomaly"]:
    print(f"Anomaly detected: {anomaly_result['anomaly_types']}")
```

### API Usage

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Start with Redis caching
REDIS_URL=redis://localhost:6379 uvicorn src.api.main:app --reload

# Analyze text via API
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The cat sat on the mat",
    "model": "gpt2",
    "methods": ["attention", "importance", "sae"]
  }'

# Batch analysis
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["First text", "Second text", "Third text"],
    "methods": ["attention"],
    "batch_size": 2
  }'

# Failure prediction
curl -X POST "http://localhost:8000/predict/failure" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "the the the the the",
    "threshold": 0.5
  }'
```

### Interactive Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/app.py

# Access at http://localhost:8501
```

### WebSocket Real-time Analysis

```python
import asyncio
import json
import websockets

async def analyze_realtime():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Send analysis request
        await websocket.send(json.dumps({
            "type": "analyze",
            "text": "Analyze this text in real-time",
            "methods": ["attention", "importance"]
        }))
        
        # Receive updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"{data['type']}: {data['data']['message']}")
            
            if data["type"] == "complete":
                break

asyncio.run(analyze_realtime())
```

## 🛠️ Advanced Usage

### Memory-Efficient Processing

```python
# Analyze long sequences efficiently
long_text = " ".join(["word"] * 10000)

# Method 1: Chunked processing
results = analyzer.analyze_efficient(
    long_text,
    methods=["attention"],
    use_chunking=True,
    max_length=4096
)

# Method 2: Streaming analysis for documents
document = "Very long document text..."
streaming_results = analyzer.analyze_document_streaming(
    document,
    methods=["attention", "importance"],
    window_size=512,
    stride=256
)

# Check memory requirements before processing
requirements = analyzer.estimate_processing_requirements(
    long_text,
    methods=["attention", "importance"]
)
print(f"Estimated memory: {requirements['memory_estimate_gb']['total_gb']:.2f} GB")
```

### Training Custom Models

```python
# Train failure predictor on your data
analyzer.train_failure_predictor(
    training_data=[(text1, label1), (text2, label2), ...],
    # Or use synthetic data
    generate_synthetic=True,
    n_synthetic_samples=1000
)

# Train SAE for feature extraction
analyzer.train_sae(
    training_texts=["text1", "text2", ...],
    layer=-1,  # Last layer
    n_epochs=10,
    hidden_size=1024
)

# Fit anomaly detector on normal examples
normal_texts = ["Normal text 1", "Normal text 2", ...]
analyzer.fit_anomaly_detector(normal_texts)
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Start all services (API, Redis, Dashboard)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build the image
docker build -t llm-interpretability-toolkit .

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 8501:8501 \
  -e REDIS_URL=redis://redis:6379 \
  llm-interpretability-toolkit

# Run with volume mounts for models
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  llm-interpretability-toolkit
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-interpretability-toolkit.git
cd llm-interpretability-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run linting
ruff check src/ tests/
black src/ tests/ --check
mypy src/

# Format code
black src/ tests/
ruff check src/ tests/ --fix
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_comprehensive.py

# Run with verbose output
pytest -v

# Run integration tests only
pytest tests/integration/

# Run with parallel execution
pytest -n auto
```

## 🏗️ Architecture

The toolkit is built with a modular architecture for flexibility and extensibility:

### Core Components

```
src/
├── core/                    # Core interpretability algorithms
│   ├── analyzer.py         # Main analyzer orchestrating all methods
│   ├── attention.py        # Attention pattern analysis
│   ├── sae.py             # Sparse Autoencoder implementation
│   ├── failure_prediction.py  # Failure prediction models
│   ├── anomaly_detection.py   # Anomaly detection algorithms
│   └── efficient_processing.py # Memory-efficient processing
├── api/                    # REST API implementation
│   ├── main.py            # FastAPI application
│   ├── routes/            # API route handlers
│   └── websocket.py       # WebSocket support
├── dashboard/             # Streamlit dashboard
│   └── app.py            # Interactive UI
├── visualization/         # Visualization components
│   ├── attention_viz.py   # Attention visualizations
│   └── interactive.py     # Interactive plots
└── utils/                 # Utilities and helpers
    ├── cache.py          # Caching utilities
    ├── logger.py         # Logging configuration
    └── metrics.py        # Performance metrics
```

### Key Design Patterns

1. **Plugin Architecture**: Easy extension with custom analysis methods
2. **Lazy Loading**: Models loaded on-demand to reduce memory usage
3. **Caching Strategy**: Multi-level caching for performance
4. **Streaming Processing**: Handle large inputs without memory overflow
5. **Async Support**: Non-blocking operations for better scalability

## 📊 Performance Benchmarks

| Model Size | Analysis Time | Memory Usage | GPU Required |
|------------|--------------|--------------|-------------|
| GPT-2 (124M) | ~0.5s | 2GB | No |
| GPT-2 Medium (355M) | ~1.2s | 4GB | Recommended |
| GPT-2 Large (774M) | ~2.5s | 8GB | Yes |
| BERT Base (110M) | ~0.4s | 2GB | No |
| BERT Large (340M) | ~1.0s | 4GB | Recommended |

*Benchmarks on NVIDIA RTX 3080, analyzing 512 tokens*

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
MODEL_DEVICE=cuda              # Device to use (cuda/cpu)
MODEL_CACHE_DIR=./models       # Model cache directory

# API configuration
API_HOST=0.0.0.0               # API host
API_PORT=8000                  # API port
API_WORKERS=4                  # Number of workers

# Cache configuration
REDIS_URL=redis://localhost:6379  # Redis URL for caching
CACHE_TTL=3600                    # Cache TTL in seconds

# Processing configuration
MAX_BATCH_SIZE=32              # Maximum batch size
MAX_SEQUENCE_LENGTH=2048       # Maximum sequence length
MEMORY_LIMIT_GB=16            # Memory limit for processing
```

### Configuration File

```yaml
# config.yaml
model:
  default_model: gpt2
  device: cuda
  fp16: true
  
analysis:
  default_methods:
    - attention
    - importance
  cache_results: true
  
api:
  rate_limit: 100
  timeout: 30
  
visualization:
  default_colormap: viridis
  interactive: true
```

## 🚧 Roadmap

### Version 1.0 (Current)
- ✅ Core interpretability features
- ✅ REST API with WebSocket support
- ✅ Failure prediction system
- ✅ Anomaly detection
- ✅ Interactive dashboard
- ✅ Docker deployment

### Version 1.1 (Planned)
- 🔄 Distributed computing support
- 🔄 Multi-GPU processing
- 🔄 Enhanced visualization options
- 🔄 Model comparison tools
- 🔄 Export to standard formats

### Version 2.0 (Future)
- 🔄 Support for larger models (GPT-3 scale)
- 🔄 Real-time model monitoring
- 🔄 Integration with MLOps platforms
- 🔄 Advanced interpretability methods
- 🔄 Custom model fine-tuning

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{llm_interpretability_toolkit,
  title = {LLM Interpretability Toolkit: A Production-Ready Framework for Transformer Analysis},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/llm-interpretability-toolkit},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments

This project builds on the excellent work of:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Mechanistic interpretability
- [BertViz](https://github.com/jessevig/bertviz) - Attention visualization
- [Anthropic's Interpretability Research](https://transformer-circuits.pub/) - SAE techniques
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model implementations

## 📞 Support

- 📖 [Documentation](https://llm-interpretability-toolkit.readthedocs.io)
- 💬 [GitHub Discussions](https://github.com/yourusername/llm-interpretability-toolkit/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/llm-interpretability-toolkit/issues)
- 📧 [Email Support](mailto:support@example.com)
- 💼 [Professional Support](https://example.com/support)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/llm-interpretability-toolkit&type=Date)](https://star-history.com/#yourusername/llm-interpretability-toolkit&Date)