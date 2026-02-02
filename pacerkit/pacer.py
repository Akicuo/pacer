"""
Main PACER Merger class - the primary public API.

This module provides the high-level interface for running PACER merges.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from pacerkit.config import PACERConfig, load_config, load_config_from_dict
from pacerkit.core.alignment import PermutationAligner
from pacerkit.core.consensus import ConsensusEngine
from pacerkit.core.interference import InterferenceAnalyzer
from pacerkit.core.merging import DARETIESMerger
from pacerkit.core.moe import MoEBuilder, PACER_MoE_Layer
from pacerkit.utils.logging import get_logger, setup_logging
from pacerkit.utils.model_io import (
    load_models,
    save_model,
    verify_architecture_compatibility,
    get_model_architecture_info,
)
from pacerkit.utils.activation import collect_activations


class PACERMerger:
    """
    PACER: Permutation-Aligned Consensus Expert Routing Merger.
    
    This is the main class for performing PACER model merges. It handles
    the complete pipeline:
    
    1. Load and validate models
    2. Geometric alignment (Git Re-Basin)
    3. Compute Consensus Barycenter
    4. Analyze interference per layer
    5. Merge low-interference layers (DARE-TIES)
    6. Upcycle high-interference layers to MoE
    7. Save merged model
    
    Example:
        ```python
        from pacerkit import PACERMerger
        
        merger = PACERMerger([
            "model-org/model-a",
            "model-org/model-b",
        ])
        
        merged = merger.merge(
            interference_threshold=0.35,
            output_path="./merged_model"
        )
        ```
    """
    
    def __init__(
        self,
        models: Optional[List[Union[str, nn.Module]]] = None,
        config: Optional[Union[str, Path, PACERConfig, Dict[str, Any]]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the PACER Merger.
        
        Args:
            models: List of model paths/IDs or pre-loaded models
            config: Configuration file path, dict, or PACERConfig instance
            verbose: Whether to show progress output
        """
        self.verbose = verbose
        self.logger = get_logger("PACERMerger")
        
        if verbose:
            setup_logging()
        
        # Load config if provided
        if config is not None:
            if isinstance(config, (str, Path)):
                self.config = load_config(config)
            elif isinstance(config, dict):
                self.config = load_config_from_dict(config)
            else:
                self.config = config
        else:
            self.config = PACERConfig()
        
        # Handle models input
        if models is not None:
            self.config.models = models if isinstance(models[0], str) else []
            self._raw_models = models if not isinstance(models[0], str) else None
        else:
            self._raw_models = None
        
        # Will be populated during merge
        self.loaded_models: Optional[List[nn.Module]] = None
        self.aligned_models: Optional[List[nn.Module]] = None
        self.consensus_model: Optional[nn.Module] = None
        self.merged_model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None  # Tokenizer from first model
        
        # Components
        self.aligner: Optional[PermutationAligner] = None
        self.consensus_engine: Optional[ConsensusEngine] = None
        self.interference_analyzer: Optional[InterferenceAnalyzer] = None
        self.merger: Optional[DARETIESMerger] = None
        
        # Results
        self.interference_report: Dict[str, Tuple[float, str]] = {}
        self.moe_layers: Dict[str, PACER_MoE_Layer] = {}
        self.activation_report: Dict[str, Any] = {}
        self.keep_overrides: Dict[str, Dict[str, Any]] = {}
    
    def load_models(self) -> List[nn.Module]:
        """
        Load all models specified in configuration.
        
        Returns:
            List of loaded models
        """
        if self._raw_models is not None:
            self.loaded_models = self._raw_models
            if self.verbose:
                print(f"📦 Using {len(self.loaded_models)} pre-loaded models")
        else:
            if self.verbose:
                print(f"📦 Loading {len(self.config.models)} models...")
            
            self.loaded_models = load_models(
                model_paths=self.config.models,
                torch_dtype=self.config.model_config.torch_dtype,
                device_map=self.config.model_config.device_map,
                trust_remote_code=self.config.model_config.trust_remote_code,
                use_flash_attention=self.config.model_config.use_flash_attention,
                verbose=self.verbose,
            )
        
        # Load tokenizer from first model (all should be compatible)
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                if self.config.models:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.models[0],
                        trust_remote_code=self.config.model_config.trust_remote_code,
                    )
                    if self.verbose:
                        print(f"   Loaded tokenizer from: {self.config.models[0]}")
                else:
                    self.tokenizer = None
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Could not load tokenizer: {e}")
                self.tokenizer = None
        
        # Verify compatibility
        verify_architecture_compatibility(self.loaded_models)
        
        if self.verbose:
            info = get_model_architecture_info(self.loaded_models[0])
            print(f"   Architecture: {info.get('architecture', 'Unknown')}")
            print(f"   Parameters: {info['total_parameters']:,}")
        
        return self.loaded_models
    
    def align(
        self,
        models: Optional[List[nn.Module]] = None,
    ) -> List[nn.Module]:
        """
        Perform geometric alignment using Git Re-Basin.
        
        Args:
            models: Models to align (uses loaded models if None)
            
        Returns:
            List of aligned models
        """
        if models is not None:
            self.loaded_models = models
        
        if self.loaded_models is None:
            self.load_models()
        
        if self.verbose:
            print("\n🔧 Phase 1: Geometric Alignment")
        
        # First model is anchor, rest are peers
        anchor = self.loaded_models[0]
        peers = self.loaded_models[1:]
        
        self.aligner = PermutationAligner(
            anchor_model=anchor,
            peer_models=peers,
            anchor_strategy=self.config.pacer.anchor_strategy,
            verbose=self.verbose,
            num_workers=self.config.processing.num_workers,
            fast_mode=self.config.processing.fast_mode,
        )
        
        aligned_peers = self.aligner.align()
        
        # Combine anchor with aligned peers
        self.aligned_models = [anchor] + aligned_peers
        
        if self.verbose:
            for i, peer_idx in enumerate(range(len(peers))):
                quality = self.aligner.compute_alignment_quality(peer_idx)
                print(f"   Model {i+1} alignment quality: {quality:.4f}")
        
        return self.aligned_models
    
    def compute_consensus(
        self,
        aligned_models: Optional[List[nn.Module]] = None,
    ) -> nn.Module:
        """
        Compute the Consensus Barycenter.
        
        Args:
            aligned_models: Aligned models (uses internal if None)
            
        Returns:
            Consensus model
        """
        if aligned_models is not None:
            self.aligned_models = aligned_models
        
        if self.aligned_models is None:
            self.align()
        
        if self.verbose:
            print("\n📊 Phase 2: Consensus Barycenter")
        
        self.consensus_engine = ConsensusEngine(
            aligned_models=self.aligned_models,
            verbose=self.verbose,
        )
        
        self.consensus_model = self.consensus_engine.compute_consensus()
        self.consensus_engine.compute_deviation_vectors()
        
        return self.consensus_model
    
    def analyze_interference(self) -> Dict[str, Tuple[float, str]]:
        """
        Analyze interference for all layers.
        
        Returns:
            Dict mapping layer names to (score, decision) tuples
        """
        if self.consensus_engine is None:
            self.compute_consensus()
        
        if self.verbose:
            print("\n🔍 Phase 3: Interference Analysis")
        
        self.interference_analyzer = InterferenceAnalyzer(
            threshold=self.config.pacer.interference_threshold,
            verbose=self.verbose,
        )
        
        # Get param names that have deviations
        param_names = self.consensus_engine.get_parameter_names()
        
        self.interference_report = self.interference_analyzer.analyze_all_layers(
            deviation_getter=self.consensus_engine.get_layer_deviations,
            layer_names=param_names,
            filter_mlp_only=True,  # Only consider MLP layers for MoE
        )
        
        return self.interference_report
    
    def merge(
        self,
        interference_threshold: Optional[float] = None,
        top_k_experts: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Run the complete PACER merge pipeline.
        
        Args:
            interference_threshold: Override config threshold
            top_k_experts: Override config top_k
            output_path: Path to save merged model (optional)
            **kwargs: Additional config overrides
            
        Returns:
            Merged model
        """
        # Apply overrides
        if interference_threshold is not None:
            self.config.pacer.interference_threshold = interference_threshold
        if top_k_experts is not None:
            self.config.pacer.top_k_experts = top_k_experts
        
        if self.verbose:
            print("=" * 60)
            print("🚀 PACER Model Merging Pipeline")
            print("=" * 60)
            print(f"   Models: {len(self.config.models)}")
            print(f"   Interference threshold: {self.config.pacer.interference_threshold}")
            print(f"   Top-K experts: {self.config.pacer.top_k_experts}")
            print("=" * 60)
        
        # Run pipeline
        self.load_models()
        self._collect_activation_stats()
        self.align()
        self.compute_consensus()
        self.analyze_interference()
        self._prepare_keep_overrides()
        
        # Phase 4: Build merged model
        if self.verbose:
            print("\n🔨 Phase 4: Building Merged Model")
        
        self.merged_model = self._build_merged_model()
        
        # Save if path provided
        if output_path is not None:
            output_path = Path(output_path)
            self.save(output_path)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("✅ PACER Merge Complete!")
            print("=" * 60)
            self._print_summary()
        
        return self.merged_model
    
    def _build_merged_model(self) -> nn.Module:
        """
        Build the final merged model with DARE-TIES and MoE layers.
        
        Returns:
            Merged model
        """
        # Start with consensus model
        merged = copy.deepcopy(self.consensus_model)
        consensus_state = self.consensus_model.state_dict()
        merged_state = merged.state_dict()
        
        # Initialize merger and MoE builder
        self.merger = DARETIESMerger(
            dropout_rate=self.config.pacer.dropout_rate,
            verbose=False,
        )
        
        moe_builder = MoEBuilder(top_k=self.config.pacer.top_k_experts)
        keep_map = self.keep_overrides.get("param_sources", {})
        aligned_states = [
            model.state_dict() for model in (self.aligned_models or [])
        ]
        
        # Process each layer based on interference decision
        for layer_name, (score, decision) in tqdm(
            self.interference_report.items(),
            desc="Merging layers",
            disable=not self.verbose,
        ):
            try:
                deviations = self.consensus_engine.get_layer_deviations(layer_name)
                consensus_param = consensus_state[layer_name]
                
                if layer_name in keep_map:
                    src_idx = keep_map[layer_name]
                    merged_state[layer_name] = aligned_states[src_idx][layer_name]
                    continue
                
                if decision == "merge":
                    # DARE-TIES merge
                    merged_param = self.merger.merge_deviations(
                        deviations=deviations,
                        consensus_param=consensus_param,
                    )
                    merged_state[layer_name] = merged_param
                    
                elif decision == "upcycle_moe" and self.config.pacer.enable_moe_upcycle:
                    # Build MoE layer
                    # Note: Actually replacing modules requires model surgery
                    # For now, we store the MoE layers separately
                    
                    # Cluster experts if similar
                    clustered_weights, clusters = self.interference_analyzer.cluster_experts(
                        deviations,
                        self.config.pacer.expert_cluster_threshold,
                    )
                    
                    # Store MoE layer info (full replacement requires more work)
                    self.moe_layers[layer_name] = {
                        "num_experts": clustered_weights.shape[0],
                        "clusters": clusters,
                        "score": score,
                    }
                    
                    # For now, keep consensus weights
                    # Full MoE architecture conversion would happen here
                    merged_state[layer_name] = consensus_param
                    
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠ Error processing {layer_name}: {e}")
        
        # Apply activation-guided overrides to any remaining parameters
        for param_name, src_idx in keep_map.items():
            if param_name in merged_state and param_name in aligned_states[src_idx]:
                merged_state[param_name] = aligned_states[src_idx][param_name]
        
        # Load merged state
        merged.load_state_dict(merged_state)
        
        return merged

    def _collect_activation_stats(self) -> None:
        """Collect activation stats and build keep overrides if enabled."""
        if not self.config.activation.enabled:
            return
        
        if not self.config.model_prompts:
            raise ValueError(
                "Activation-based merge is enabled but no model prompts were provided."
            )
        
        if self.verbose:
            print("\n🧠 Phase 0: Activation Collection")
        
        self.activation_report = collect_activations(
            model_prompts=self.config.model_prompts,
            settings=self.config.activation,
            trust_remote_code=self.config.model_config.trust_remote_code,
        )

    def _prepare_keep_overrides(self) -> None:
        """Create keep overrides once consensus is available."""
        if not self.config.activation.enabled:
            return
        if not self.activation_report:
            return
        self.keep_overrides = self._build_keep_overrides()

    def _build_keep_overrides(self) -> Dict[str, Dict[str, Any]]:
        """
        Build parameter overrides based on activation stats.
        
        Returns:
            Dict with param_sources mapping and metadata for reporting.
        """
        keep_param_sources: Dict[str, int] = {}
        keep_param_meta: Dict[str, Dict[str, Any]] = {}
        
        model_stats = self.activation_report.get("models", {})
        if not model_stats:
            return {"param_sources": keep_param_sources, "param_meta": keep_param_meta}
        
        param_names = list(self.consensus_model.state_dict().keys()) if self.consensus_model else []
        for model_idx, model_id in enumerate(self.config.models):
            stats = model_stats.get(model_id, {})
            layer_scores = stats.get("layers", {})
            expert_scores = stats.get("experts", {})
            
            selected_layers = _select_top_items(
                layer_scores,
                self.config.activation.keep_top_layers,
                self.config.activation.min_activation_score,
            )
            for layer_name, score in selected_layers.items():
                _update_param_overrides(
                    keep_param_sources,
                    keep_param_meta,
                    param_names,
                    layer_name,
                    model_idx,
                    model_id,
                    score,
                    target_type="layer",
                )
            
            for layer_name, experts in expert_scores.items():
                selected_experts = _select_top_items(
                    experts,
                    self.config.activation.keep_top_experts_per_layer,
                    self.config.activation.min_activation_score,
                )
                for expert_id, score in selected_experts.items():
                    expert_prefix = f"{layer_name}.experts.{expert_id}"
                    _update_param_overrides(
                        keep_param_sources,
                        keep_param_meta,
                        param_names,
                        expert_prefix,
                        model_idx,
                        model_id,
                        score,
                        target_type="expert",
                        target_id=str(expert_id),
                    )
        
        return {"param_sources": keep_param_sources, "param_meta": keep_param_meta}

    def _build_activation_summary(self) -> Dict[str, Any]:
        """Build a concise activation summary for reports and README."""
        if not self.config.activation.enabled:
            return {"enabled": False}
        
        keep_sources = self.keep_overrides.get("param_sources", {})
        keep_meta = self.keep_overrides.get("param_meta", {})
        keep_by_model: Dict[str, int] = {model_id: 0 for model_id in self.config.models}
        for param_name, meta in keep_meta.items():
            model_id = meta.get("model_id")
            if model_id in keep_by_model:
                keep_by_model[model_id] += 1
        
        return {
            "enabled": True,
            "backend": self.activation_report.get("backend", self.config.activation.backend),
            "settings": self.activation_report.get("settings", {}),
            "model_prompts": [
                {"hf_id": cfg.hf_id, "p_prompts": cfg.p_prompts}
                for cfg in self.config.model_prompts
            ],
            "keep_parameters": len(keep_sources),
            "keep_by_model": keep_by_model,
        }
    
    def _print_summary(self) -> None:
        """Print merge summary."""
        summary = self.interference_analyzer.get_summary()
        
        print(f"\n📈 Merge Summary:")
        print(f"   Total layers analyzed: {summary['total_layers']}")
        print(f"   Merged layers (DARE-TIES): {summary['merge_layers']}")
        print(f"   MoE layers: {summary['moe_layers']}")
        print(f"   Average interference: {summary['avg_interference']:.4f}")
        
        if self.moe_layers:
            total_experts = sum(info['num_experts'] for info in self.moe_layers.values())
            print(f"\n🧩 MoE Details:")
            print(f"   Layers converted: {len(self.moe_layers)}")
            print(f"   Total experts: {total_experts}")
    
    def save(
        self,
        output_path: Union[str, Path],
        save_format: Optional[str] = None,
        push_to_hub: Optional[bool] = None,
        hub_repo: Optional[str] = None,
        hub_token: Optional[str] = None,
        private: Optional[bool] = None,
    ) -> str:
        """
        Save the merged model and optionally upload to HuggingFace Hub.
        
        Args:
            output_path: Output directory
            save_format: Override config save format
            push_to_hub: Override config push_to_hub setting
            hub_repo: Override config hub_repo
            hub_token: HuggingFace API token
            private: Whether to create a private repository
            
        Returns:
            Path to saved model (or Hub URL if pushed)
        """
        import datetime
        
        if self.merged_model is None:
            raise RuntimeError("No merged model to save. Run merge() first.")
        
        output_path = Path(output_path)
        format_to_use = save_format or self.config.output.save_format
        do_push = push_to_hub if push_to_hub is not None else self.config.output.push_to_hub
        repo = hub_repo or self.config.output.hub_repo
        token = hub_token or self.config.output.hub_token
        is_private = private if private is not None else self.config.output.private
        
        # Add timestamp to output path if configured
        if self.config.output.add_timestamp and not output_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path.parent / f"{output_path.name}_{timestamp}"
        
        # Create output directory structure
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "logs").mkdir(exist_ok=True)
        
        if self.verbose:
            print(f"\n💾 Saving to: {output_path}")
        
        # Build merge config for model card
        activation_summary = self._build_activation_summary()
        merge_config = {
            "project_name": self.config.project_name,
            "models": self.config.models,
            "pacer": {
                "interference_threshold": self.config.pacer.interference_threshold,
                "top_k_experts": self.config.pacer.top_k_experts,
            },
            "summary": self.interference_analyzer.get_summary() if self.interference_analyzer else {},
            "activation": activation_summary,
        }
        
        # Save model with optional Hub upload
        result = save_model(
            model=self.merged_model,
            output_path=output_path,
            save_format=format_to_use,
            tokenizer=getattr(self, 'tokenizer', None),  # Include tokenizer if loaded
            push_to_hub=do_push,
            hub_repo=repo,
            hub_token=token,
            private=is_private,
            merge_config=merge_config if self.config.output.generate_model_card else None,
            verbose=self.verbose,
        )
        
        # Save merge report
        self._save_merge_report(output_path / "merge_report.json")
        
        if self.verbose:
            print(f"✓ Model saved!")
            if do_push and repo:
                print(f"🌐 Published to: https://huggingface.co/{repo}")
        
        return result
    
    def _save_merge_report(self, path: Path) -> None:
        """Save detailed merge report as JSON."""
        import json
        
        report = {
            "config": {
                "project_name": self.config.project_name,
                "models": self.config.models,
                "interference_threshold": self.config.pacer.interference_threshold,
                "top_k_experts": self.config.pacer.top_k_experts,
            },
            "summary": self.interference_analyzer.get_summary() if self.interference_analyzer else {},
            "activation": self._build_activation_summary(),
            "layer_decisions": {
                name: {"score": score, "decision": decision}
                for name, (score, decision) in self.interference_report.items()
            },
            "moe_layers": {
                name: {"num_experts": info["num_experts"], "score": info["score"]}
                for name, info in self.moe_layers.items()
            },
            "keep_overrides": self.keep_overrides.get("param_meta", {}),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    
    def get_interference_report(self) -> Dict[str, Any]:
        """
        Get detailed interference analysis report.
        
        Returns:
            Dict with per-layer interference data
        """
        return {
            "summary": self.interference_analyzer.get_summary() if self.interference_analyzer else {},
            "layers": self.interference_report,
            "high_interference": self.interference_analyzer.get_high_interference_layers(10) if self.interference_analyzer else [],
        }


def _select_top_items(
    scores: Dict[str, float],
    top_k: Optional[int],
    min_score: float,
) -> Dict[str, float]:
    filtered = {k: v for k, v in scores.items() if v >= min_score}
    if top_k is None:
        return filtered
    return dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:top_k])


def _update_param_overrides(
    keep_param_sources: Dict[str, int],
    keep_param_meta: Dict[str, Dict[str, Any]],
    param_names: List[str],
    prefix: str,
    model_idx: int,
    model_id: str,
    score: float,
    target_type: str,
    target_id: Optional[str] = None,
) -> None:
    for param_name in param_names:
        if not param_name.startswith(prefix + "."):
            continue
        existing = keep_param_meta.get(param_name)
        if existing and existing.get("score", 0.0) >= score:
            continue
        keep_param_sources[param_name] = model_idx
        keep_param_meta[param_name] = {
            "model_id": model_id,
            "score": score,
            "target_type": target_type,
            "target_id": target_id,
            "prefix": prefix,
        }


def merge_models(
    models: List[str],
    output_path: str,
    interference_threshold: float = 0.35,
    top_k_experts: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Convenience function to merge models with PACER.
    
    Args:
        models: List of model paths/IDs
        output_path: Where to save merged model
        interference_threshold: Threshold for MoE upcycling
        top_k_experts: Number of experts per token
        **kwargs: Additional options
        
    Returns:
        Merged model
    """
    merger = PACERMerger(models=models)
    return merger.merge(
        interference_threshold=interference_threshold,
        top_k_experts=top_k_experts,
        output_path=output_path,
        **kwargs,
    )
