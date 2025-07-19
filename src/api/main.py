"""FastAPI application for LLM Interpretability Toolkit"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core import InterpretabilityAnalyzer, Config
from ..utils.logger import get_logger, setup_logging
from .websocket import websocket_endpoint

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global analyzer instance
analyzer: Optional[InterpretabilityAnalyzer] = None
config = Config.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    global analyzer
    logger.info("Starting LLM Interpretability Toolkit API")
    
    # Initialize analyzer with default model
    analyzer = InterpretabilityAnalyzer(
        model_name=config.model.name or "gpt2",
        config=config,
    )
    
    logger.info(f"Loaded model: {analyzer.model_name}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="LLM Interpretability Toolkit API",
    description="REST API for analyzing and visualizing attention patterns in transformer models",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint"""
    
    text: Union[str, List[str]] = Field(
        ...,
        description="Text or list of texts to analyze",
        example="The cat sat on the mat"
    )
    model: Optional[str] = Field(
        None,
        description="Model to use for analysis",
        example="gpt2"
    )
    methods: Optional[List[str]] = Field(
        None,
        description="Analysis methods to apply",
        example=["attention", "importance"]
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional options for analysis methods"
    )


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch analysis endpoint"""
    
    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        example=["First text", "Second text", "Third text"]
    )
    model: Optional[str] = Field(
        None,
        description="Model to use for analysis",
        example="gpt2"
    )
    methods: Optional[List[str]] = Field(
        None,
        description="Analysis methods to apply",
        example=["attention", "importance"]
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional options for analysis methods"
    )
    batch_size: int = Field(
        8,
        description="Batch size for processing",
        ge=1,
        le=32
    )


class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint"""
    
    results: Dict[str, Any] = Field(
        ...,
        description="Analysis results"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata about the analysis"
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process request in seconds"
    )


class FailurePredictionRequest(BaseModel):
    """Request model for failure prediction endpoint"""
    
    text: Union[str, List[str]] = Field(
        ...,
        description="Text to analyze for failure prediction"
    )
    threshold: float = Field(
        0.5,
        description="Threshold for failure prediction",
        ge=0.0,
        le=1.0
    )


class FailurePredictionResponse(BaseModel):
    """Response model for failure prediction endpoint"""
    
    failure_probability: float = Field(
        ...,
        description="Probability of model failure",
        ge=0.0,
        le=1.0
    )
    prediction: str = Field(
        ...,
        description="Risk level (high_risk/low_risk)"
    )
    indicators: List[str] = Field(
        ...,
        description="List of failure indicators detected"
    )
    confidence: float = Field(
        ...,
        description="Confidence in the prediction",
        ge=0.0,
        le=1.0
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation"
    )


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint"""
    
    model_name: str
    architecture: str
    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    device: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    
    status: str
    model_loaded: bool
    uptime: float


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    
    enabled: bool
    backend: str
    stats: Dict[str, Any]


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Interpretability Toolkit API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=analyzer is not None,
        uptime=time.time(),  # In production, track actual uptime
    )


@app.get("/models", response_model=List[str])
async def list_models():
    """List available models"""
    from ..core.model_wrapper import ModelWrapper
    return ModelWrapper.list_supported_models()


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=analyzer.model_name,
        architecture=analyzer.model_wrapper.model.config.model_type,
        num_layers=analyzer.model_wrapper.get_num_layers(),
        num_heads=analyzer.model_wrapper.get_num_attention_heads(),
        hidden_size=analyzer.model_wrapper.get_hidden_size(),
        vocab_size=analyzer.model_wrapper.get_vocab_size(),
        device=str(analyzer.device),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text using specified methods"""
    start_time = time.time()
    
    try:
        # Use global analyzer or create new one if different model requested
        current_analyzer = analyzer
        if request.model and request.model != analyzer.model_name:
            logger.info(f"Loading model: {request.model}")
            current_analyzer = InterpretabilityAnalyzer(
                model_name=request.model,
                config=config,
            )
        
        # Perform analysis
        results = current_analyzer.analyze(
            text=request.text,
            methods=request.methods,
            **(request.options or {})
        )
        
        # Convert tensors to lists for JSON serialization
        results = _serialize_results(results)
        
        processing_time = time.time() - start_time
        
        return AnalyzeResponse(
            results=results,
            metadata={
                "model": current_analyzer.model_name,
                "methods": request.methods or ["attention", "importance"],
                "text_length": len(request.text) if isinstance(request.text, str) else len(request.text[0]),
            },
            processing_time=processing_time,
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/failure", response_model=FailurePredictionResponse)
async def predict_failure(request: FailurePredictionRequest):
    """Predict model failure probability"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analyzer.predict_failure_probability(
            text=request.text,
            threshold=request.threshold,
        )
        
        return FailurePredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Failure prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model(model_name: str):
    """Load a different model"""
    global analyzer
    
    try:
        logger.info(f"Loading model: {model_name}")
        analyzer = InterpretabilityAnalyzer(
            model_name=model_name,
            config=config,
        )
        
        return {
            "message": f"Model {model_name} loaded successfully",
            "model_info": {
                "name": analyzer.model_name,
                "num_layers": analyzer.model_wrapper.get_num_layers(),
                "num_heads": analyzer.model_wrapper.get_num_attention_heads(),
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyze multiple texts in batches for better performance"""
    start_time = time.time()
    
    try:
        # Use global analyzer or create new one if different model requested
        current_analyzer = analyzer
        if request.model and request.model != analyzer.model_name:
            logger.info(f"Loading model: {request.model}")
            current_analyzer = InterpretabilityAnalyzer(
                model_name=request.model,
                config=config,
            )
        
        # Process texts in batches
        all_results = []
        batch_size = request.batch_size
        
        for i in range(0, len(request.texts), batch_size):
            batch_texts = request.texts[i:i + batch_size]
            
            # Analyze batch
            batch_results = current_analyzer.analyze(
                text=batch_texts,
                methods=request.methods,
                **(request.options or {})
            )
            
            # Split results by text
            for j, text in enumerate(batch_texts):
                text_results = {}
                
                # Extract results for this specific text
                for method in (request.methods or ["attention", "importance"]):
                    if method in batch_results:
                        method_data = batch_results[method]
                        
                        # Handle batch dimension in results
                        if method == "attention":
                            # Extract attention for this text
                            text_results[method] = {
                                "patterns": method_data["patterns"][:, j:j+1],
                                "tokens": [method_data["tokens"][j]],
                                "shape": method_data["shape"].copy(),
                                "entropy": {
                                    k: v[j] if len(v.shape) > 0 else v
                                    for k, v in method_data.get("entropy", {}).items()
                                }
                            }
                            text_results[method]["shape"]["batch_size"] = 1
                        else:
                            # For non-batch methods, use as-is
                            text_results[method] = method_data
                
                all_results.append({
                    "text": text,
                    "results": text_results,
                    "index": i + j
                })
        
        processing_time = time.time() - start_time
        
        # Serialize results
        serialized_results = _serialize_results({
            "batch_results": all_results,
            "total_texts": len(request.texts),
            "batch_size": batch_size,
            "methods": request.methods or ["attention", "importance"],
        })
        
        return {
            "results": serialized_results,
            "metadata": {
                "model": current_analyzer.model_name,
                "total_texts": len(request.texts),
                "batch_size": batch_size,
                "methods": request.methods or ["attention", "importance"],
            },
            "processing_time": processing_time,
        }
    
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    stats = analyzer.cache_manager.get_stats()
    
    return CacheStatsResponse(
        enabled=stats.get("enabled", False),
        backend=stats.get("backend", "none"),
        stats=stats
    )


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached results"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    success = analyzer.cache_manager.clear()
    
    if success:
        return {"message": "Cache cleared successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.websocket("/ws")
async def websocket_analysis(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis updates"""
    if analyzer is None:
        await websocket.close(code=1003, reason="Model not loaded")
        return
    
    await websocket_endpoint(websocket, analyzer)


def _serialize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensors to JSON-serializable format"""
    import torch
    import numpy as np
    
    def serialize_value(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [serialize_value(v) for v in value]
        else:
            return value
    
    return serialize_value(results)


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.api.log_level.lower(),
    )