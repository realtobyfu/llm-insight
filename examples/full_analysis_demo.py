"""
Full demonstration of LLM Interpretability Toolkit capabilities

This script shows how to:
1. Analyze attention patterns
2. Compute token importance
3. Train and use Sparse Autoencoders
4. Predict model failures
5. Create visualizations
6. Use the API programmatically
"""

import asyncio
import json
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch

from src.core import InterpretabilityAnalyzer


def demonstrate_basic_analysis():
    """Demonstrate basic analysis capabilities"""
    print("\n=== Basic Analysis Demo ===")
    
    # Initialize analyzer
    analyzer = InterpretabilityAnalyzer(model_name="distilgpt2")
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog. This is a classic pangram sentence."
    
    # Perform comprehensive analysis
    results = analyzer.analyze(
        text,
        methods=["attention", "importance", "head_patterns"],
    )
    
    print(f"\nAnalyzed text: '{text}'")
    print(f"Model: {results['metadata']['model_name']}")
    print(f"Number of layers: {results['metadata']['num_layers']}")
    print(f"Number of heads: {results['metadata']['num_heads']}")
    
    # Show attention entropy
    mean_entropy = results["attention"]["entropy"]["mean_entropy_per_layer"].mean()
    print(f"\nMean attention entropy: {mean_entropy:.3f}")
    
    # Show token importance
    tokens = results["importance"]["token_importance"]["tokens"]
    importance = results["importance"]["token_importance"]["importance_mean"]
    
    print("\nTop 5 most important tokens:")
    top_indices = torch.argsort(torch.tensor(importance), descending=True)[:5]
    for idx in top_indices:
        print(f"  - '{tokens[idx]}': {importance[idx]:.3f}")
    
    # Show head patterns
    patterns = results["head_patterns"]["pattern_summary"]
    print(f"\nIdentified attention patterns: {patterns}")
    
    return analyzer, results


def demonstrate_sae_analysis():
    """Demonstrate SAE feature extraction"""
    print("\n=== SAE Feature Analysis Demo ===")
    
    analyzer = InterpretabilityAnalyzer(model_name="distilgpt2")
    
    # Train SAE on sample data
    print("\nTraining SAE on sample texts...")
    training_texts = [
        "Machine learning is transforming the world.",
        "Natural language processing enables many applications.",
        "Deep learning models can understand complex patterns.",
        "Artificial intelligence is becoming more sophisticated.",
        "Neural networks learn from large amounts of data.",
    ] * 5  # Repeat for more training data
    
    history = analyzer.train_sae(training_texts, layer=-1, n_epochs=3)
    print(f"Final training loss: {history['total_loss'][-1]:.4f}")
    
    # Analyze text with SAE
    test_text = "Understanding how neural networks make decisions is important."
    results = analyzer.analyze(test_text, methods=["sae"])
    
    sae_data = results["sae"]
    print(f"\nNumber of learned features: {sae_data['sae_config']['n_features']}")
    print(f"Number of dead features: {sae_data['num_dead_features']}")
    
    # Show top features for first position
    if sae_data["decompositions"]:
        first_pos = sae_data["decompositions"][0]
        print(f"\nTop features at position 0:")
        for feat_idx, feat_val in zip(first_pos["top_features"][:3], first_pos["top_values"][:3]):
            print(f"  - Feature {feat_idx}: {feat_val:.3f}")
    
    return analyzer, sae_data


def demonstrate_failure_prediction():
    """Demonstrate failure prediction capabilities"""
    print("\n=== Failure Prediction Demo ===")
    
    analyzer = InterpretabilityAnalyzer(model_name="distilgpt2")
    
    test_cases = [
        "This is a normal sentence with proper structure.",
        "the the the the the the the the the",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "Random words elephant computer flying purple banana logic.",
    ]
    
    print("\nAnalyzing texts for potential failures:")
    for text in test_cases:
        result = analyzer.predict_failure_probability(text)
        print(f"\nText: '{text[:50]}...'")
        print(f"Failure probability: {result['failure_probability']:.2%}")
        print(f"Risk level: {result['prediction']}")
        if result['indicators']:
            print(f"Indicators: {', '.join(result['indicators'])}")


def demonstrate_visualization():
    """Demonstrate visualization capabilities"""
    print("\n=== Visualization Demo ===")
    
    analyzer = InterpretabilityAnalyzer(model_name="distilgpt2")
    
    # Analyze text
    text = "Attention mechanisms help models focus on relevant information."
    results = analyzer.analyze(text, methods=["attention", "importance"])
    
    # Create visualizations
    output_dir = Path("examples/outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Attention heatmap
    print("\nCreating attention heatmap...")
    fig_attention = analyzer.visualize_attention(
        results["attention"],
        save_path=output_dir / "attention_heatmap.png",
        layer=0,
        head=0
    )
    plt.close(fig_attention)
    
    # 2. Token importance
    print("Creating token importance plot...")
    fig_importance = analyzer.visualize_token_importance(
        results["importance"],
        save_path=output_dir / "token_importance.png"
    )
    plt.close(fig_importance)
    
    # 3. Interactive visualization
    print("Creating interactive attention visualization...")
    fig_interactive = analyzer.visualize_attention(
        results["attention"],
        save_path=output_dir / "attention_interactive.html",
        interactive=True
    )
    
    print(f"\nVisualizations saved to {output_dir}")


async def demonstrate_api_usage():
    """Demonstrate API usage"""
    print("\n=== API Usage Demo ===")
    print("\nNote: Make sure the API server is running:")
    print("  uvicorn src.api.main:app --reload")
    
    base_url = "http://localhost:8000"
    
    try:
        # Check health
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("\n✓ API is running")
        else:
            print("\n✗ API is not responding")
            return
    except:
        print("\n✗ Could not connect to API")
        return
    
    # Example 1: Basic analysis
    print("\n1. Basic analysis:")
    response = requests.post(
        f"{base_url}/analyze",
        json={
            "text": "Hello world from the API!",
            "methods": ["attention", "importance"]
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Processing time: {data['processing_time']:.2f}s")
        print(f"   Methods used: {data['metadata']['methods']}")
    
    # Example 2: Batch analysis
    print("\n2. Batch analysis:")
    response = requests.post(
        f"{base_url}/analyze/batch",
        json={
            "texts": [
                "First text to analyze",
                "Second text to analyze",
                "Third text to analyze"
            ],
            "methods": ["attention"],
            "batch_size": 2
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Total texts: {data['metadata']['total_texts']}")
        print(f"   Batch size: {data['metadata']['batch_size']}")
        print(f"   Processing time: {data['processing_time']:.2f}s")
    
    # Example 3: Cache statistics
    print("\n3. Cache statistics:")
    response = requests.get(f"{base_url}/cache/stats")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Cache enabled: {data['enabled']}")
        print(f"   Backend: {data['backend']}")
        if data['enabled']:
            print(f"   Keys cached: {data['stats'].get('keys', 'N/A')}")


def demonstrate_websocket_usage():
    """Demonstrate WebSocket real-time updates"""
    print("\n=== WebSocket Demo ===")
    print("\nWebSocket example (requires running API):")
    
    example_code = '''
import asyncio
import json
import websockets

async def analyze_with_updates():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Send analysis request
        await websocket.send(json.dumps({
            "type": "analyze",
            "text": "Real-time analysis of this text",
            "methods": ["attention", "importance", "sae"]
        }))
        
        # Receive updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "progress":
                print(f"Progress: {data['data']['progress']}% - {data['data']['message']}")
            elif data["type"] == "complete":
                print("Analysis complete!")
                break
            elif data["type"] == "error":
                print(f"Error: {data['data']['message']}")
                break

# Run: asyncio.run(analyze_with_updates())
'''
    
    print(example_code)


def main():
    """Run all demonstrations"""
    print("LLM Interpretability Toolkit - Full Demo")
    print("=" * 50)
    
    # Run demos
    demonstrate_basic_analysis()
    demonstrate_sae_analysis()
    demonstrate_failure_prediction()
    demonstrate_visualization()
    
    # API demos (require running server)
    asyncio.run(demonstrate_api_usage())
    demonstrate_websocket_usage()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the examples/outputs directory for visualizations.")


if __name__ == "__main__":
    main()