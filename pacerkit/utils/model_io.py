"""
Model I/O utilities for loading and saving models.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.
    
    Args:
        dtype_str: String representation (e.g., "bfloat16", "float16")
        
    Returns:
        Corresponding torch.dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)


def load_model(
    model_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
) -> nn.Module:
    """
    Load a model from HuggingFace or local path.
    
    Args:
        model_path: HuggingFace model ID or local path
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        use_flash_attention: Whether to use Flash Attention 2
        
    Returns:
        Loaded model
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError(
            "transformers is required for model loading. "
            "Install with: pip install transformers"
        )
    
    dtype = get_torch_dtype(torch_dtype)
    
    # Load config first to check architecture
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    
    # Build kwargs
    kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    
    # Add flash attention if supported
    if use_flash_attention:
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass  # Flash attention not available
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    return model


def load_models(
    model_paths: List[str],
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
    verbose: bool = True,
) -> List[nn.Module]:
    """
    Load multiple models.
    
    Args:
        model_paths: List of HuggingFace model IDs or local paths
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        use_flash_attention: Whether to use Flash Attention 2
        verbose: Whether to show progress
        
    Returns:
        List of loaded models
    """
    from tqdm import tqdm
    
    models = []
    iterator = tqdm(model_paths, desc="Loading models") if verbose else model_paths
    
    for path in iterator:
        if verbose:
            print(f"   Loading: {path}")
        
        model = load_model(
            path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            use_flash_attention=use_flash_attention,
        )
        models.append(model)
    
    return models


def save_model(
    model: nn.Module,
    output_path: Union[str, Path],
    save_format: Literal["safetensors", "pytorch"] = "safetensors",
    tokenizer: Optional[Any] = None,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
    hub_token: Optional[str] = None,
    private: bool = False,
    model_card: Optional[str] = None,
    merge_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> str:
    """
    Save a model to disk and optionally upload to HuggingFace Hub.
    
    Args:
        model: Model to save
        output_path: Output directory path
        save_format: Format to save in ("safetensors" or "pytorch")
        tokenizer: Optional tokenizer to save with model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo: Hub repository name (required if push_to_hub)
        hub_token: HuggingFace API token (uses cached token if None)
        private: Whether to create a private repository
        model_card: Custom model card content (auto-generated if None)
        merge_config: Merge configuration for model card generation
        verbose: Whether to print progress
        
    Returns:
        Path to saved model (or Hub URL if pushed)
    """
    import json
    import datetime
    
    output_path = Path(output_path)
    
    # Create output directory with timestamp if needed
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"📁 Created output directory: {output_path}")
    
    # Determine save method based on model type
    if hasattr(model, "save_pretrained"):
        # HuggingFace model
        safe_serialization = (save_format == "safetensors")
        
        if verbose:
            print(f"💾 Saving model to: {output_path}")
            print(f"   Format: {save_format}")
        
        model.save_pretrained(
            output_path,
            safe_serialization=safe_serialization,
        )
        
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)
            if verbose:
                print(f"   Tokenizer saved")
        
        # Save merge configuration
        if merge_config is not None:
            config_path = output_path / "merge_config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(merge_config, f, indent=2, default=str)
            if verbose:
                print(f"   Merge config saved to: {config_path}")
        
        # Generate and save model card
        readme_path = output_path / "README.md"
        if model_card is not None:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(model_card)
        elif merge_config is not None:
            # Auto-generate model card
            card_content = _generate_model_card(merge_config)
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(card_content)
            if verbose:
                print(f"   Model card generated")
        
        # Push to HuggingFace Hub
        if push_to_hub:
            if hub_repo is None:
                raise ValueError("hub_repo is required when push_to_hub=True")
            
            hub_url = push_to_huggingface_hub(
                model=model,
                tokenizer=tokenizer,
                repo_id=hub_repo,
                token=hub_token,
                private=private,
                model_path=output_path,
                verbose=verbose,
            )
            return hub_url
    else:
        # Generic PyTorch model
        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                save_path = output_path / "model.safetensors"
                save_file(model.state_dict(), save_path)
            except ImportError:
                save_path = output_path / "model.pt"
                torch.save(model.state_dict(), save_path)
        else:
            save_path = output_path / "model.pt"
            torch.save(model.state_dict(), save_path)
        
        if verbose:
            print(f"💾 Saved PyTorch model to: {save_path}")
    
    return str(output_path)


def push_to_huggingface_hub(
    model: nn.Module,
    repo_id: str,
    tokenizer: Optional[Any] = None,
    token: Optional[str] = None,
    private: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    commit_message: str = "Upload PACER merged model",
    verbose: bool = True,
) -> str:
    """
    Upload a model to HuggingFace Hub.
    
    Args:
        model: Model to upload
        repo_id: Repository ID (e.g., "username/model-name")
        tokenizer: Optional tokenizer to upload
        token: HuggingFace API token (uses cached if None)
        private: Whether to create a private repository
        model_path: Local path containing model files (if already saved)
        commit_message: Commit message for the upload
        verbose: Whether to print progress
        
    Returns:
        URL of the uploaded model
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for Hub uploads. "
            "Install with: pip install huggingface_hub"
        )
    
    if verbose:
        print(f"\n🚀 Uploading to HuggingFace Hub: {repo_id}")
    
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
        )
        if verbose:
            print(f"   Repository created/verified: {repo_id}")
    except Exception as e:
        if verbose:
            print(f"   Repository check: {e}")
    
    # Upload model
    if model_path is not None:
        # Upload from folder
        model_path = Path(model_path)
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )
    elif hasattr(model, "push_to_hub"):
        # Use model's push_to_hub method
        model.push_to_hub(repo_id, token=token, private=private)
        if tokenizer is not None:
            tokenizer.push_to_hub(repo_id, token=token, private=private)
    else:
        raise ValueError(
            "Either model_path must be provided or model must have push_to_hub method"
        )
    
    hub_url = f"https://huggingface.co/{repo_id}"
    
    if verbose:
        print(f"✅ Model uploaded successfully!")
        print(f"   URL: {hub_url}")
    
    return hub_url


def _generate_model_card(merge_config: Dict[str, Any]) -> str:
    """
    Generate a model card for a PACER merged model.
    
    Args:
        merge_config: Merge configuration dictionary
        
    Returns:
        Model card content as markdown string
    """
    import datetime
    
    models = merge_config.get("models", [])
    project_name = merge_config.get("project_name", "PACER Merged Model")
    threshold = merge_config.get("pacer", {}).get("interference_threshold", 0.35)
    top_k = merge_config.get("pacer", {}).get("top_k_experts", 2)
    
    merged_layers = merge_config.get("summary", {}).get("merge_layers", "N/A")
    moe_layers = merge_config.get("summary", {}).get("moe_layers", "N/A")
    
    model_list = "\n".join([f"- `{m}`" for m in models])
    
    card = f"""---
library_name: transformers
tags:
- pacer
- model-merging
- merged-model
- moe
license: apache-2.0
---

# {project_name}

This model was created using **PACER (Permutation-Aligned Consensus Expert Routing)**.

## Model Details

**Merge Type:** PACER (Base-Free, Interference-Aware)

**Source Models:**
{model_list}

**Merge Configuration:**
- Interference Threshold: `{threshold}`
- Top-K Experts: `{top_k}`
- Merged Layers: `{merged_layers}`
- MoE Layers: `{moe_layers}`

## How PACER Works

PACER is a novel model merging framework that:
1. **Aligns models geometrically** using Git Re-Basin
2. **Computes a Consensus Barycenter** as a synthetic base
3. **Analyzes interference** per layer
4. **Merges low-interference layers** using DARE-TIES
5. **Upcycles high-interference layers** to Mixture-of-Experts

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{project_name}")
tokenizer = AutoTokenizer.from_pretrained("{project_name}")

# Use the model
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
```

## Created With

[PacerKit](https://github.com/yourusername/pacerkit) - PACER Model Merging Framework

**Created:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    return card


def create_output_folder(
    base_path: Union[str, Path],
    project_name: str,
    add_timestamp: bool = True,
) -> Path:
    """
    Create an output folder for merged models.
    
    Args:
        base_path: Base directory for output
        project_name: Name of the project/merge
        add_timestamp: Whether to add timestamp to folder name
        
    Returns:
        Path to created output folder
    """
    import datetime
    
    base_path = Path(base_path)
    
    if add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{project_name}_{timestamp}"
    else:
        folder_name = project_name
    
    output_path = base_path / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)
    
    return output_path


def get_model_architecture_info(model: nn.Module) -> Dict[str, Any]:
    """
    Extract architecture information from a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dict with architecture details
    """
    info = {
        "model_type": type(model).__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "layers": {},
    }
    
    # Count layer types
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in info["layers"]:
            info["layers"][module_type] = 0
        info["layers"][module_type] += 1
    
    # Check for HuggingFace config
    if hasattr(model, "config"):
        config = model.config
        info["hidden_size"] = getattr(config, "hidden_size", None)
        info["num_hidden_layers"] = getattr(config, "num_hidden_layers", None)
        info["num_attention_heads"] = getattr(config, "num_attention_heads", None)
        info["vocab_size"] = getattr(config, "vocab_size", None)
        info["architecture"] = getattr(config, "architectures", [None])[0]
    
    return info


def verify_architecture_compatibility(
    models: List[nn.Module],
) -> bool:
    """
    Verify that multiple models have compatible architectures.
    
    Args:
        models: List of models to check
        
    Returns:
        True if compatible, raises ValueError otherwise
    """
    if len(models) < 2:
        return True
    
    reference = get_model_architecture_info(models[0])
    
    for i, model in enumerate(models[1:], 1):
        current = get_model_architecture_info(model)
        
        # Check critical dimensions
        if reference.get("hidden_size") != current.get("hidden_size"):
            raise ValueError(
                f"Hidden size mismatch: Model 0 has {reference.get('hidden_size')}, "
                f"Model {i} has {current.get('hidden_size')}"
            )
        
        if reference.get("num_hidden_layers") != current.get("num_hidden_layers"):
            raise ValueError(
                f"Layer count mismatch: Model 0 has {reference.get('num_hidden_layers')}, "
                f"Model {i} has {current.get('num_hidden_layers')}"
            )
    
    return True
