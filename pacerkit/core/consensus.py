"""
Consensus Barycenter computation for base-free merging.

This module computes the Fréchet Mean (Barycenter) of aligned models
and calculates deviation vectors for each model.
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


class ConsensusEngine:
    """
    Compute the Consensus Barycenter and deviation vectors.
    
    The Consensus Barycenter serves as the synthetic "base model" for
    merging when no actual base model is available. Deviation vectors
    represent how each model differs from this consensus.
    
    Attributes:
        models: List of aligned models
        consensus_model: The computed barycenter model
        deviation_vectors: Per-model deviation from consensus
    """
    
    def __init__(
        self,
        aligned_models: List[nn.Module],
        verbose: bool = True,
    ):
        """
        Initialize the consensus engine.
        
        Args:
            aligned_models: List of models that have been aligned
            verbose: Whether to show progress
        """
        if len(aligned_models) < 2:
            raise ValueError("At least 2 models required for consensus")
        
        self.models = aligned_models
        self.num_models = len(aligned_models)
        self.verbose = verbose
        
        # Will be populated by compute_consensus
        self.consensus_model: Optional[nn.Module] = None
        self.deviation_vectors: Dict[str, torch.Tensor] = {}
        
    def compute_consensus(self) -> nn.Module:
        """
        Compute the Consensus Barycenter (Fréchet Mean) of aligned models.
        
        This creates a new model with weights that are the arithmetic mean
        of all input model weights. This serves as the synthetic base model.
        
        Returns:
            The consensus model with averaged weights
        """
        if self.verbose:
            print("📊 Computing Consensus Barycenter...")
        
        # Deep copy first model as template
        self.consensus_model = copy.deepcopy(self.models[0])
        
        # Get state dicts from all models
        state_dicts = [m.state_dict() for m in self.models]
        consensus_state = self.consensus_model.state_dict()
        
        # Average each parameter
        param_names = list(consensus_state.keys())
        iterator = tqdm(param_names, desc="Averaging weights") if self.verbose else param_names
        
        with torch.no_grad():
            for name in iterator:
                # Stack parameters from all models
                try:
                    stacked = torch.stack([sd[name].float() for sd in state_dicts])
                    # Compute arithmetic mean
                    consensus_state[name] = torch.mean(stacked, dim=0).to(
                        consensus_state[name].dtype
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠ Skipping {name}: {e}")
        
        # Load averaged state dict
        self.consensus_model.load_state_dict(consensus_state)
        
        if self.verbose:
            print("✓ Consensus Barycenter computed!")
        
        return self.consensus_model
    
    def compute_deviation_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Compute deviation vectors for each model from the consensus.
        
        Deviation vector: Δ_k = θ̂_k - θ_consensus
        
        These vectors represent how each model differs from the group average.
        High magnitude values indicate specialization or noise.
        
        Returns:
            Dict mapping "param_name_modelidx" to deviation tensors
        """
        if self.consensus_model is None:
            raise RuntimeError("Must call compute_consensus() first")
        
        if self.verbose:
            print("📐 Computing Deviation Vectors...")
        
        consensus_state = self.consensus_model.state_dict()
        self.deviation_vectors = {}
        
        for model_idx, model in enumerate(self.models):
            model_state = model.state_dict()
            
            for name, consensus_param in consensus_state.items():
                if name not in model_state:
                    continue
                
                model_param = model_state[name]
                
                # Compute deviation: model - consensus
                deviation = (model_param.float() - consensus_param.float())
                
                # Store with model index suffix
                key = f"{name}__model_{model_idx}"
                self.deviation_vectors[key] = deviation
        
        if self.verbose:
            print(f"✓ Computed {len(self.deviation_vectors)} deviation vectors")
        
        return self.deviation_vectors
    
    def get_layer_deviations(
        self,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Get stacked deviation vectors for a specific layer.
        
        Args:
            layer_name: Name of the layer (e.g., "model.layers.0.mlp.gate_proj.weight")
            
        Returns:
            Tensor of shape (N, *param_shape) with deviations from each model
        """
        deviations = []
        
        for model_idx in range(self.num_models):
            key = f"{layer_name}__model_{model_idx}"
            if key in self.deviation_vectors:
                deviations.append(self.deviation_vectors[key])
        
        if not deviations:
            raise KeyError(f"No deviation vectors found for layer: {layer_name}")
        
        return torch.stack(deviations)
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names from consensus model."""
        if self.consensus_model is None:
            raise RuntimeError("Must call compute_consensus() first")
        return list(self.consensus_model.state_dict().keys())
    
    def compute_deviation_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics about deviation vectors.
        
        Returns:
            Dict with per-parameter statistics (mean, std, max deviation)
        """
        if not self.deviation_vectors:
            raise RuntimeError("Must call compute_deviation_vectors() first")
        
        stats = {}
        param_names = set()
        
        # Extract unique parameter names
        for key in self.deviation_vectors.keys():
            param_name = key.rsplit("__model_", 1)[0]
            param_names.add(param_name)
        
        for param_name in param_names:
            try:
                deviations = self.get_layer_deviations(param_name)
                
                # Compute stats
                norms = torch.norm(deviations.view(deviations.shape[0], -1), dim=1)
                
                stats[param_name] = {
                    "mean_deviation_norm": norms.mean().item(),
                    "std_deviation_norm": norms.std().item(),
                    "max_deviation_norm": norms.max().item(),
                    "min_deviation_norm": norms.min().item(),
                }
            except Exception:
                continue
        
        return stats
