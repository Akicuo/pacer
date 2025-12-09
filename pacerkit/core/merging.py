"""
DARE-TIES merging for low-interference layers.

This module implements the DARE (Drop And REscale) and TIES (TrIm, Elect, Sign)
merging strategies for combining weights when interference is low.
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


class DARETIESMerger:
    """
    Merge model weights using DARE-TIES strategy.
    
    DARE: Randomly drop small deviations with probability p, then rescale.
    TIES: Trim redundant parameters, elect sign by magnitude, resolve conflicts.
    
    This merger is used for layers with low interference where simple
    averaging (with some cleanup) is effective.
    """
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        density: float = 0.9,
        sign_strategy: Literal["sum", "max", "mean"] = "sum",
        verbose: bool = True,
    ):
        """
        Initialize the DARE-TIES merger.
        
        Args:
            dropout_rate: Probability of dropping a deviation value
            density: Fraction of parameters to keep (for trimming)
            sign_strategy: How to resolve sign conflicts
            verbose: Whether to show progress
        """
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
        if not 0.0 <= density <= 1.0:
            raise ValueError("density must be between 0.0 and 1.0")
        
        self.dropout_rate = dropout_rate
        self.density = density
        self.sign_strategy = sign_strategy
        self.verbose = verbose
    
    def _trim_by_magnitude(
        self,
        deviations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trim deviations by magnitude, keeping only top-k% by absolute value.
        
        Args:
            deviations: Tensor of shape (N, *param_shape)
            
        Returns:
            Trimmed deviations with small values zeroed
        """
        N = deviations.shape[0]
        flat = deviations.view(N, -1).abs()
        
        # Compute threshold per model
        k = int(self.density * flat.shape[1])
        if k < 1:
            k = 1
        
        # Get top-k threshold for each model
        thresholds, _ = torch.topk(flat, k, dim=1)
        threshold_vals = thresholds[:, -1].view(N, *([1] * (deviations.dim() - 1)))
        
        # Zero out values below threshold
        mask = deviations.abs() >= threshold_vals
        return deviations * mask
    
    def _elect_sign(
        self,
        deviations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Elect the dominant sign for each parameter position.
        
        Args:
            deviations: Tensor of shape (N, *param_shape)
            
        Returns:
            Sign tensor with values in {-1, 0, +1}
        """
        if self.sign_strategy == "sum":
            # Sign is determined by sum of deviations
            total = torch.sum(deviations, dim=0)
            return torch.sign(total)
        
        elif self.sign_strategy == "max":
            # Sign is determined by largest magnitude
            abs_devs = deviations.abs()
            max_idx = torch.argmax(abs_devs, dim=0)
            # Gather the deviation with max magnitude
            max_dev = torch.gather(
                deviations, 0,
                max_idx.unsqueeze(0)
            ).squeeze(0)
            return torch.sign(max_dev)
        
        else:  # mean
            mean_dev = torch.mean(deviations, dim=0)
            return torch.sign(mean_dev)
    
    def _apply_dare_dropout(
        self,
        deviations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply DARE dropout to deviations.
        
        Randomly drops values and rescales remaining to compensate.
        
        Args:
            deviations: Tensor of shape (N, *param_shape)
            
        Returns:
            Dropout-applied and rescaled deviations
        """
        if self.dropout_rate <= 0:
            return deviations
        
        # Create dropout mask
        mask = torch.bernoulli(
            torch.full_like(deviations, 1 - self.dropout_rate)
        )
        
        # Apply mask and rescale
        scale = 1.0 / (1.0 - self.dropout_rate)
        return deviations * mask * scale
    
    def merge_deviations(
        self,
        deviations: torch.Tensor,
        consensus_param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge deviation vectors using DARE-TIES strategy.
        
        Args:
            deviations: Deviation vectors of shape (N, *param_shape)
            consensus_param: Base consensus parameter to add merged deviation to
            
        Returns:
            Merged parameter tensor
        """
        # Step 1: Trim by magnitude
        trimmed = self._trim_by_magnitude(deviations)
        
        # Step 2: Apply DARE dropout
        dropped = self._apply_dare_dropout(trimmed)
        
        # Step 3: Elect sign
        elected_sign = self._elect_sign(dropped)
        
        # Step 4: Sum only values that agree with elected sign
        # Mask out values that conflict with elected sign
        sign_mask = (torch.sign(dropped) == elected_sign) | (dropped == 0)
        aligned = dropped * sign_mask
        
        # Step 5: Average the aligned deviations
        merged_deviation = torch.mean(aligned, dim=0)
        
        # Step 6: Add to consensus
        merged = consensus_param.float() + merged_deviation
        
        return merged.to(consensus_param.dtype)
    
    def merge_layer(
        self,
        layer_name: str,
        deviations: torch.Tensor,
        consensus_param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge a single layer's parameters.
        
        Args:
            layer_name: Name of the layer (for logging)
            deviations: Deviation vectors for this layer
            consensus_param: Consensus parameter
            
        Returns:
            Merged parameter tensor
        """
        return self.merge_deviations(deviations, consensus_param)


class SimpleMerger:
    """
    Simple averaging merger for comparison/fallback.
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize simple merger.
        
        Args:
            weights: Optional weights for each model (uniform if None)
        """
        self.weights = weights
    
    def merge(
        self,
        tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge tensors using weighted average.
        
        Args:
            tensors: List of tensors to merge
            
        Returns:
            Merged tensor
        """
        stacked = torch.stack(tensors)
        
        if self.weights is None:
            return torch.mean(stacked, dim=0)
        
        weights = torch.tensor(
            self.weights,
            device=stacked.device,
            dtype=stacked.dtype,
        ).view(-1, *([1] * (stacked.dim() - 1)))
        
        return torch.sum(stacked * weights, dim=0)
