"""
Configuration loading and validation for PacerKit.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml


@dataclass
class OutputConfig:
    """Output configuration settings."""
    path: str = "./merged_model"
    save_format: Literal["safetensors", "pytorch"] = "safetensors"
    push_to_hub: bool = False
    hub_repo: Optional[str] = None
    hub_token: Optional[str] = None
    private: bool = False
    add_timestamp: bool = True
    generate_model_card: bool = True


@dataclass
class PACERSettings:
    """PACER algorithm settings."""
    interference_threshold: float = 0.35
    top_k_experts: int = 2
    dropout_rate: float = 0.1
    anchor_strategy: Literal["first", "lowest_magnitude", "random"] = "first"
    enable_moe_upcycle: bool = True
    expert_cluster_threshold: float = 0.9


@dataclass
class ModelSettings:
    """Model loading configuration."""
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_flash_attention: bool = True


@dataclass
class VisionSettings:
    """Vision-specific settings for ViT merging."""
    enable_token_merging: bool = False
    token_merge_ratio: float = 0.3
    merge_before_routing: bool = True


@dataclass
class ProcessingSettings:
    """Processing options."""
    alignment_batch_size: int = 1000
    show_progress: bool = True
    verbose: bool = True
    num_workers: int = 4  # Parallel workers for alignment
    fast_mode: bool = True  # Use approximations for large layers


@dataclass
class PACERConfig:
    """
    Complete PACER configuration.
    
    Attributes:
        project_name: Name of the merge project
        models: List of model paths or HuggingFace model IDs
        output: Output configuration
        pacer: PACER algorithm settings
        model_config: Model loading settings
        vision: Vision-specific settings
        processing: Processing options
    """
    project_name: str = "pacer-merge"
    models: List[str] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    pacer: PACERSettings = field(default_factory=PACERSettings)
    model_config: ModelSettings = field(default_factory=ModelSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if len(self.models) < 2:
            raise ValueError("At least 2 models are required for merging")
        
        if not 0.0 <= self.pacer.interference_threshold <= 1.0:
            raise ValueError("interference_threshold must be between 0.0 and 1.0")
        
        if self.pacer.top_k_experts < 1:
            raise ValueError("top_k_experts must be at least 1")
        
        if not 0.0 <= self.pacer.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
        
        if self.vision.enable_token_merging:
            if not 0.0 <= self.vision.token_merge_ratio <= 1.0:
                raise ValueError("token_merge_ratio must be between 0.0 and 1.0")


def _dict_to_dataclass(data: Dict[str, Any], cls: type) -> Any:
    """Convert a dictionary to a dataclass instance."""
    if not data:
        return cls()
    
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(config_path: Union[str, Path]) -> PACERConfig:
    """
    Load PACER configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        PACERConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raise ValueError(f"Empty configuration file: {config_path}")
    
    # Parse nested configs
    output = _dict_to_dataclass(raw_config.get("output", {}), OutputConfig)
    pacer = _dict_to_dataclass(raw_config.get("pacer", {}), PACERSettings)
    model_config = _dict_to_dataclass(raw_config.get("model_config", {}), ModelSettings)
    vision = _dict_to_dataclass(raw_config.get("vision", {}), VisionSettings)
    processing = _dict_to_dataclass(raw_config.get("processing", {}), ProcessingSettings)
    
    config = PACERConfig(
        project_name=raw_config.get("project_name", "pacer-merge"),
        models=raw_config.get("models", []),
        output=output,
        pacer=pacer,
        model_config=model_config,
        vision=vision,
        processing=processing,
    )
    
    config.validate()
    return config


def save_config(config: PACERConfig, path: Union[str, Path]) -> None:
    """
    Save PACER configuration to a YAML file.
    
    Args:
        config: PACERConfig instance
        path: Output path for the YAML file
    """
    path = Path(path)
    
    # Convert dataclasses to dictionaries
    from dataclasses import asdict
    data = asdict(config)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
