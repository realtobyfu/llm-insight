# LLM Interpretability Toolkit - API Reference

## Table of Contents
1. [REST API](#rest-api)
2. [Python API](#python-api)
3. [WebSocket API](#websocket-api)
4. [CLI Reference](#cli-reference)

## REST API

The toolkit provides a comprehensive REST API for integration with other applications.

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, the API does not require authentication. In production, implement appropriate security measures.

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": 1234.56
}
```

#### List Models
```http
GET /models
```

**Response:**
```json
[
  "gpt2",
  "gpt2-medium",
  "bert-base-uncased",
  "distilbert-base-uncased"
]
```

#### Model Information
```http
GET /model/info
```

**Response:**
```json
{
  "model_name": "gpt2",
  "architecture": "gpt2",
  "num_layers": 12,
  "num_heads": 12,
  "hidden_size": 768,
  "vocab_size": 50257,
  "device": "cuda"
}
```

#### Analyze Text
```http
POST /analyze
```

**Request Body:**
```json
{
  "text": "The cat sat on the mat",
  "model": "gpt2",
  "methods": ["attention", "importance", "sae"],
  "options": {
    "layer": 0,
    "head": 0
  }
}
```

**Response:**
```json
{
  "results": {
    "attention": {
      "patterns": [...],
      "tokens": [["The", "cat", "sat", "on", "the", "mat"]],
      "shape": {
        "num_layers": 12,
        "batch_size": 1,
        "num_heads": 12,
        "seq_length": 6
      },
      "entropy": {
        "mean_entropy_per_layer": [1.2, 1.3, ...],
        "entropy_by_position": [0.8, 0.9, ...]
      }
    },
    "importance": {
      "token_importance": {
        "tokens": ["The", "cat", "sat", "on", "the", "mat"],
        "importance_mean": [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]
      },
      "head_importance": [[0.2, 0.3, ...], ...]
    }
  },
  "metadata": {
    "model": "gpt2",
    "methods": ["attention", "importance"],
    "text_length": 20
  },
  "processing_time": 0.523
}
```

#### Batch Analysis
```http
POST /analyze/batch
```

**Request Body:**
```json
{
  "texts": [
    "First text to analyze",
    "Second text to analyze",
    "Third text to analyze"
  ],
  "methods": ["attention"],
  "batch_size": 2
}
```

**Response:**
```json
{
  "results": {
    "batch_results": [
      {
        "index": 0,
        "text": "First text to analyze",
        "results": {...}
      },
      ...
    ],
    "total_texts": 3,
    "batch_size": 2
  },
  "metadata": {
    "model": "gpt2",
    "total_texts": 3,
    "batch_size": 2
  },
  "processing_time": 1.234
}
```

#### Failure Prediction
```http
POST /predict/failure
```

**Request Body:**
```json
{
  "text": "the the the the the",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "failure_probability": 0.85,
  "prediction": "high_risk",
  "indicators": ["low_entropy", "attention_collapse"],
  "confidence": 0.9,
  "explanation": "Potential issues detected: Model attention is overly focused on specific tokens; Many attention heads are collapsing to single positions"
}
```

#### Cache Statistics
```http
GET /cache/stats
```

**Response:**
```json
{
  "enabled": true,
  "backend": "redis",
  "stats": {
    "keys": 42,
    "memory_used": "12.3MB",
    "hits": 123,
    "misses": 45
  }
}
```

#### Clear Cache
```http
POST /cache/clear
```

**Response:**
```json
{
  "message": "Cache cleared successfully"
}
```

## Python API

### Installation
```python
from llm_interpretability import InterpretabilityAnalyzer
```

### Basic Usage

#### Initialize Analyzer
```python
# Create analyzer
analyzer = InterpretabilityAnalyzer(
    model_name="gpt2",
    device="cuda",  # or "cpu"
    cache_dir="./models"
)
```

#### Analyze Text
```python
# Basic analysis
results = analyzer.analyze(
    text="The quick brown fox jumps over the lazy dog",
    methods=["attention", "importance", "sae", "head_patterns"]
)

# Access results
attention_patterns = results["attention"]["patterns"]
token_importance = results["importance"]["token_importance"]
sae_features = results["sae"]["feature_stats"]
```

#### Visualize Results
```python
# Create attention visualization
fig = analyzer.visualize_attention(
    results["attention"],
    layer=0,
    head=0,
    save_path="attention.png"
)

# Create interactive visualization
interactive_fig = analyzer.visualize_attention(
    results["attention"],
    interactive=True,
    save_path="attention.html"
)

# Visualize token importance
importance_fig = analyzer.visualize_token_importance(
    results["importance"],
    save_path="importance.png"
)
```

### Advanced Features

#### Failure Prediction
```python
# Train failure predictor
train_results = analyzer.train_failure_predictor(
    generate_synthetic=True,
    n_synthetic_samples=200
)
print(f"Test accuracy: {train_results['test_accuracy']:.2%}")

# Predict failures
failure_result = analyzer.predict_failure_probability(
    "the the the the the",
    threshold=0.5
)
print(f"Failure probability: {failure_result['failure_probability']:.2%}")

# Save/load trained model
analyzer.save_failure_predictor("failure_model.pkl")
analyzer.load_failure_predictor("failure_model.pkl")
```

#### Anomaly Detection
```python
# Fit anomaly detector on normal examples
normal_texts = [
    "This is a normal sentence.",
    "Another regular text example.",
    # ... more examples
]
analyzer.fit_anomaly_detector(normal_texts)

# Detect anomalies
anomaly_result = analyzer.detect_anomalies(
    "aaaaaaaaaaaaaaaaaaa",
    check_attention=True,
    check_activations=True
)

if anomaly_result["is_anomaly"]:
    print(f"Anomaly detected: {anomaly_result['anomaly_types']}")
```

#### Memory-Efficient Processing
```python
# Analyze long text efficiently
long_text = " ".join(["word"] * 10000)

# Method 1: Chunked processing
results = analyzer.analyze_efficient(
    long_text,
    methods=["attention"],
    use_chunking=True,
    max_length=4096
)

# Method 2: Streaming analysis
document = "Very long document text..."
streaming_results = analyzer.analyze_document_streaming(
    document,
    methods=["attention", "importance"],
    window_size=512,
    stride=256,
    aggregate_results=True
)

# Estimate resource requirements
requirements = analyzer.estimate_processing_requirements(
    long_text,
    methods=["attention", "importance"]
)
print(f"Estimated memory: {requirements['memory_estimate_gb']['total_gb']:.2f} GB")
print(f"Recommendations: {requirements['recommendations']}")
```

#### Sparse Autoencoder (SAE)
```python
# Train SAE on custom data
training_texts = [
    "Text for training SAE",
    "Another training example",
    # ... more texts
]

history = analyzer.train_sae(
    training_texts,
    layer=-1,  # Last layer
    n_epochs=10,
    batch_size=32
)

# Analyze with trained SAE
sae_results = analyzer.analyze(
    "New text to analyze",
    methods=["sae"]
)

# Access SAE features
top_features = sae_results["sae"]["feature_stats"]["top_features"]
dead_features = sae_results["sae"]["dead_features"]
```

## WebSocket API

The toolkit provides WebSocket support for real-time analysis updates.

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Message Types

#### Analysis Request
```json
{
  "type": "analyze",
  "text": "Text to analyze",
  "methods": ["attention", "importance"],
  "options": {}
}
```

#### Progress Updates
```json
{
  "type": "progress",
  "data": {
    "message": "Running attention analysis...",
    "progress": 50,
    "current_method": "attention"
  }
}
```

#### Results
```json
{
  "type": "complete",
  "data": {
    "message": "Analysis complete",
    "progress": 100,
    "results": {...}
  }
}
```

### Example Client
```python
import asyncio
import json
import websockets

async def analyze_with_websocket():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Send analysis request
        await websocket.send(json.dumps({
            "type": "analyze",
            "text": "The quick brown fox",
            "methods": ["attention", "importance"]
        }))
        
        # Receive updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            print(f"{data['type']}: {data['data']}")
            
            if data["type"] == "complete":
                break

# Run
asyncio.run(analyze_with_websocket())
```

## CLI Reference

### Installation
```bash
pip install -e .
```

### Commands

#### Analyze Text
```bash
# Basic analysis
llm-interpret analyze "The cat sat on the mat" --model gpt2

# With specific methods
llm-interpret analyze "Text to analyze" \
  --model bert-base-uncased \
  --methods attention importance \
  --output results.json \
  --format json
```

#### Predict Failure
```bash
llm-interpret predict-failure "the the the the" \
  --model gpt2 \
  --threshold 0.5
```

#### List Models
```bash
llm-interpret list-models
```

#### Start API Server
```bash
llm-interpret serve --host 0.0.0.0 --port 8000 --reload
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model to use | gpt2 |
| `--methods` | Analysis methods | attention, importance |
| `--output` | Output file | None (stdout) |
| `--format` | Output format | table |
| `--log-level` | Logging level | INFO |

## Error Handling

All API endpoints return standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

Error responses include detail:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently not implemented. In production, consider adding rate limiting to prevent abuse.

## Examples Repository

For more examples, see the `examples/` directory in the repository:

- `examples/basic_usage.py`: Simple analysis examples
- `examples/full_analysis_demo.py`: Comprehensive demonstration
- `examples/api_client.py`: REST API client examples
- `examples/streaming_analysis.py`: Long document processing
- `examples/custom_model.py`: Using custom models

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/llm-interpretability-toolkit/issues
- Documentation: https://llm-interpretability-toolkit.readthedocs.io
- Email: support@example.com