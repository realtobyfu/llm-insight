"""Command-line interface for LLM Interpretability Toolkit"""

import json
from pathlib import Path
from typing import List, Optional

import click
import torch
from rich.console import Console
from rich.table import Table

from .core import InterpretabilityAnalyzer
from .utils.logger import setup_logging

console = Console()


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def cli(log_level: str):
    """LLM Interpretability Toolkit CLI"""
    setup_logging(level=log_level)


@cli.command()
@click.argument("text")
@click.option("--model", default="gpt2", help="Model to use for analysis")
@click.option("--methods", "-m", multiple=True, default=["attention", "importance"], help="Analysis methods")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--format", type=click.Choice(["json", "table"]), default="table", help="Output format")
def analyze(text: str, model: str, methods: List[str], output: Optional[str], format: str):
    """Analyze text using specified methods"""
    console.print(f"[bold blue]Analyzing text with model: {model}[/bold blue]")
    
    # Initialize analyzer
    analyzer = InterpretabilityAnalyzer(model_name=model)
    
    # Perform analysis
    with console.status("[bold green]Running analysis..."):
        results = analyzer.analyze(text, methods=list(methods))
    
    # Format output
    if format == "json":
        # Convert tensors to lists for JSON serialization
        json_results = _convert_for_json(results)
        output_text = json.dumps(json_results, indent=2)
        
        if output:
            Path(output).write_text(output_text)
            console.print(f"[green]Results saved to: {output}[/green]")
        else:
            console.print(output_text)
    
    else:  # table format
        # Display results in table format
        _display_results_table(results, text)
        
        if output:
            console.print(f"[yellow]Table format cannot be saved to file. Use --format json for file output.[/yellow]")


@cli.command()
@click.argument("text")
@click.option("--model", default="gpt2", help="Model to use for analysis")
@click.option("--threshold", default=0.5, help="Failure prediction threshold")
def predict_failure(text: str, model: str, threshold: float):
    """Predict model failure probability"""
    console.print(f"[bold blue]Predicting failure probability with model: {model}[/bold blue]")
    
    # Initialize analyzer
    analyzer = InterpretabilityAnalyzer(model_name=model)
    
    # Predict failure
    with console.status("[bold green]Analyzing..."):
        result = analyzer.predict_failure_probability(text, threshold=threshold)
    
    # Display results
    table = Table(title="Failure Prediction Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Failure Probability", f"{result['failure_probability']:.2%}")
    table.add_row("Risk Level", result['prediction'])
    table.add_row("Confidence", f"{result['confidence']:.2%}")
    table.add_row("Indicators", ", ".join(result['indicators']) if result['indicators'] else "None")
    
    console.print(table)
    console.print(f"\n[italic]{result['explanation']}[/italic]")


@cli.command()
def list_models():
    """List available models"""
    from .core.model_wrapper import ModelWrapper
    
    models = ModelWrapper.list_supported_models()
    
    table = Table(title="Supported Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Provider", style="green")
    
    for model in models:
        provider = model.split("/")[0] if "/" in model else "HuggingFace"
        table.add_row(model, provider)
    
    console.print(table)


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the API server"""
    import uvicorn
    
    console.print(f"[bold green]Starting API server on {host}:{port}[/bold green]")
    console.print(f"[blue]Documentation available at: http://{host}:{port}/docs[/blue]")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


def _convert_for_json(obj):
    """Convert objects for JSON serialization"""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    else:
        return obj


def _display_results_table(results: dict, text: str):
    """Display analysis results in table format"""
    # Metadata table
    if "metadata" in results:
        meta_table = Table(title="Model Information")
        meta_table.add_column("Property", style="cyan")
        meta_table.add_column("Value", style="magenta")
        
        for key, value in results["metadata"].items():
            meta_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(meta_table)
        console.print()
    
    # Token importance
    if "importance" in results and "token_importance" in results["importance"]:
        imp_data = results["importance"]["token_importance"]
        tokens = imp_data["tokens"]
        importance = torch.tensor(imp_data["importance_mean"])
        
        imp_table = Table(title="Token Importance")
        imp_table.add_column("Token", style="cyan")
        imp_table.add_column("Importance", style="magenta")
        
        for token, score in zip(tokens, importance):
            imp_table.add_row(token, f"{score:.4f}")
        
        console.print(imp_table)
        console.print()
    
    # Entropy statistics
    if "attention" in results and "entropy" in results["attention"]:
        entropy = results["attention"]["entropy"]
        
        entropy_table = Table(title="Attention Entropy Statistics")
        entropy_table.add_column("Layer", style="cyan")
        entropy_table.add_column("Mean Entropy", style="magenta")
        
        mean_entropy = torch.tensor(entropy["mean_entropy_per_layer"])
        for i, value in enumerate(mean_entropy):
            entropy_table.add_row(f"Layer {i}", f"{value:.4f}")
        
        console.print(entropy_table)


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()