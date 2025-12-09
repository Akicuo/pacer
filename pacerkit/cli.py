"""
Command-line interface for PacerKit.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import click

from pacerkit import __version__


@click.group()
@click.version_option(version=__version__, prog_name="pacerkit")
def main():
    """
    PacerKit - PACER Model Merging Framework
    
    Merge multiple models using Permutation-Aligned Consensus Expert Routing.
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML configuration file"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output path for merged model (overrides config)"
)
@click.option(
    "--threshold", "-t",
    type=float,
    help="Interference threshold (overrides config)"
)
@click.option(
    "--top-k", "-k",
    type=int,
    help="Top-K experts for MoE layers (overrides config)"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress output"
)
def merge(config: str, output: Optional[str], threshold: Optional[float], 
          top_k: Optional[int], quiet: bool):
    """
    Merge models using a configuration file.
    
    Example:
        pacerkit merge --config configs/qwen_coder_merge.yaml
    """
    from pacerkit import PACERMerger
    from pacerkit.config import load_config
    
    try:
        cfg = load_config(config)
        
        merger = PACERMerger(config=cfg, verbose=not quiet)
        
        merge_kwargs = {}
        if threshold is not None:
            merge_kwargs["interference_threshold"] = threshold
        if top_k is not None:
            merge_kwargs["top_k_experts"] = top_k
        if output is not None:
            merge_kwargs["output_path"] = output
        else:
            merge_kwargs["output_path"] = cfg.output.path
        
        merger.merge(**merge_kwargs)
        
        if not quiet:
            click.echo(click.style("\n✓ Merge completed successfully!", fg="green"))
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--models", "-m",
    required=True,
    multiple=True,
    help="Model paths or HuggingFace IDs (specify multiple times)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="interference_report.json",
    help="Output path for analysis report"
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=0.35,
    help="Interference threshold for classification"
)
def analyze(models: List[str], output: str, threshold: float):
    """
    Analyze interference between models without merging.
    
    Example:
        pacerkit analyze -m model1 -m model2 --output report.json
    """
    from pacerkit import PACERMerger
    
    try:
        click.echo(f"Analyzing {len(models)} models...")
        
        merger = PACERMerger(models=list(models))
        merger.config.pacer.interference_threshold = threshold
        
        # Run partial pipeline
        merger.load_models()
        merger.align()
        merger.compute_consensus()
        merger.analyze_interference()
        
        # Get report
        report = merger.get_interference_report()
        
        # Save report
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        click.echo(click.style(f"\n✓ Analysis saved to: {output_path}", fg="green"))
        
        # Print summary
        summary = report.get("summary", {})
        click.echo(f"\nSummary:")
        click.echo(f"  Total layers: {summary.get('total_layers', 0)}")
        click.echo(f"  Would merge: {summary.get('merge_layers', 0)}")
        click.echo(f"  Would upcycle: {summary.get('moe_layers', 0)}")
        click.echo(f"  Avg interference: {summary.get('avg_interference', 0):.4f}")
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="pacer_config.yaml",
    help="Output path for config file"
)
@click.option(
    "--models", "-m",
    multiple=True,
    help="Model paths to include in config"
)
def init(output: str, models: List[str]):
    """
    Generate a starter configuration file.
    
    Example:
        pacerkit init --output my_config.yaml -m model1 -m model2
    """
    from pacerkit.config import PACERConfig, save_config
    
    try:
        config = PACERConfig(
            project_name="my-pacer-merge",
            models=list(models) if models else [
                "path/to/model_a",
                "path/to/model_b",
            ],
        )
        
        save_config(config, output)
        
        click.echo(click.style(f"✓ Config saved to: {output}", fg="green"))
        click.echo("\nEdit the file to customize your merge, then run:")
        click.echo(f"  pacerkit merge --config {output}")
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument("model_path")
def info(model_path: str):
    """
    Show information about a model.
    
    Example:
        pacerkit info meta-llama/Llama-2-7b-hf
    """
    from pacerkit.utils.model_io import load_model, get_model_architecture_info
    
    try:
        click.echo(f"Loading model: {model_path}")
        
        model = load_model(
            model_path,
            device_map="cpu",  # Don't need GPU for info
            torch_dtype="float32",
        )
        
        info = get_model_architecture_info(model)
        
        click.echo(f"\nModel Information:")
        click.echo(f"  Type: {info['model_type']}")
        click.echo(f"  Architecture: {info.get('architecture', 'Unknown')}")
        click.echo(f"  Total parameters: {info['total_parameters']:,}")
        click.echo(f"  Hidden size: {info.get('hidden_size', 'N/A')}")
        click.echo(f"  Layers: {info.get('num_hidden_layers', 'N/A')}")
        click.echo(f"  Attention heads: {info.get('num_attention_heads', 'N/A')}")
        click.echo(f"  Vocab size: {info.get('vocab_size', 'N/A')}")
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
