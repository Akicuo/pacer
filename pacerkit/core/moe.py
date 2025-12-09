"""
Mixture-of-Experts layer construction and Zero-Shot Router.

This module handles the conversion of high-interference layers to MoE
and implements the data-free routing mechanism using Subspace Projection.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroShotRouter(nn.Module):
    """
    Zero-shot router using Subspace Projection Affinity.
    
    This router requires no training. It routes tokens based on how much
    each expert would modify the input compared to the consensus.
    
    Routing score: R_k(x) = ||Δ_k · x||₂
    
    Tokens are routed to experts where they would cause the largest deviation
    from consensus behavior.
    """
    
    def __init__(
        self,
        expert_deviations: torch.Tensor,
        top_k: int = 2,
        normalize_scores: bool = True,
    ):
        """
        Initialize the zero-shot router.
        
        Args:
            expert_deviations: Tensor of shape (N_experts, d_out, d_in)
                             representing weight deviations from consensus
            top_k: Number of experts to activate per token
            normalize_scores: Whether to normalize routing scores
        """
        super().__init__()
        
        self.top_k = top_k
        self.normalize_scores = normalize_scores
        self.num_experts = expert_deviations.shape[0]
        
        # Validate top_k
        if top_k > self.num_experts:
            self.top_k = self.num_experts
        
        # Compute input sensitivity prototypes
        # Sum absolute magnitudes along output dimension to get input sensitivity
        # Shape: (N_experts, d_in)
        prototypes = torch.sum(torch.abs(expert_deviations.float()), dim=1)
        
        if normalize_scores:
            prototypes = F.normalize(prototypes, dim=1)
        
        # Register as buffer (not a trainable parameter)
        self.register_buffer("prototypes", prototypes)
        
        # Store full deviations for alternative routing methods
        self.register_buffer("expert_deviations", expert_deviations.float())
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input tokens to experts.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_in)
               or (batch, d_in) for non-sequential data
               
        Returns:
            Tuple of:
                - router_weights: Softmax weights for top-k experts
                  Shape: (batch, [seq_len,] top_k)
                - expert_indices: Indices of selected experts
                  Shape: (batch, [seq_len,] top_k)
        """
        # Handle both sequential and non-sequential inputs
        input_shape = x.shape
        is_sequential = x.dim() == 3
        
        if is_sequential:
            batch, seq_len, d_in = x.shape
            x_flat = x.view(batch * seq_len, d_in)
        else:
            x_flat = x
        
        # Compute alignment scores with prototypes
        # Score = dot product with prototype (higher = more relevant)
        scores = torch.matmul(x_flat, self.prototypes.t())  # (B*S, N_experts)
        
        # Select top-k experts
        router_logits, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        
        # Softmax for gating weights
        router_weights = F.softmax(router_logits, dim=-1)
        
        # Reshape back if sequential
        if is_sequential:
            router_weights = router_weights.view(batch, seq_len, self.top_k)
            expert_indices = expert_indices.view(batch, seq_len, self.top_k)
        
        return router_weights, expert_indices
    
    def compute_routing_scores(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full routing scores for all experts (for analysis).
        
        Args:
            x: Input tensor of shape (batch, [seq_len,] d_in)
            
        Returns:
            Scores tensor of shape (batch, [seq_len,] num_experts)
        """
        if x.dim() == 3:
            batch, seq_len, d_in = x.shape
            x_flat = x.view(batch * seq_len, d_in)
            scores = torch.matmul(x_flat, self.prototypes.t())
            return scores.view(batch, seq_len, self.num_experts)
        else:
            return torch.matmul(x, self.prototypes.t())


class PACER_MoE_Layer(nn.Module):
    """
    Mixture-of-Experts layer for PACER.
    
    This replaces a dense linear layer with multiple experts and
    a zero-shot router. Only top-k experts are activated per token.
    """
    
    def __init__(
        self,
        experts: List[nn.Linear],
        router: ZeroShotRouter,
        consensus_layer: Optional[nn.Linear] = None,
    ):
        """
        Initialize the MoE layer.
        
        Args:
            experts: List of nn.Linear modules (one per expert)
            router: ZeroShotRouter instance
            consensus_layer: Optional consensus layer for residual connection
        """
        super().__init__()
        
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.consensus_layer = consensus_layer
        self.num_experts = len(experts)
        self.top_k = router.top_k
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_in)
               or (batch, d_in)
               
        Returns:
            Output tensor of same shape as expert output
        """
        # Get routing decisions
        router_weights, expert_indices = self.router(x)
        
        # Handle shapes
        is_sequential = x.dim() == 3
        if is_sequential:
            batch, seq_len, d_in = x.shape
            x_flat = x.view(batch * seq_len, d_in)
            weights_flat = router_weights.view(batch * seq_len, self.top_k)
            indices_flat = expert_indices.view(batch * seq_len, self.top_k)
        else:
            batch = x.shape[0]
            x_flat = x
            weights_flat = router_weights
            indices_flat = expert_indices
        
        # Get output dimension from first expert
        d_out = self.experts[0].out_features
        
        # Initialize output
        output = torch.zeros(x_flat.shape[0], d_out, device=x.device, dtype=x.dtype)
        
        # Process each expert
        for k in range(self.top_k):
            k_weights = weights_flat[:, k].unsqueeze(-1)  # (B*S, 1)
            k_indices = indices_flat[:, k]  # (B*S,)
            
            # Process tokens for each expert
            for expert_idx in range(self.num_experts):
                mask = (k_indices == expert_idx)
                
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    expert_weight = k_weights[mask]
                    
                    output[mask] += expert_output * expert_weight
        
        # Reshape back if sequential
        if is_sequential:
            output = output.view(batch, seq_len, d_out)
        
        return output


class MoEBuilder:
    """
    Build MoE layers from expert weights and deviation vectors.
    """
    
    def __init__(
        self,
        top_k: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize the MoE builder.
        
        Args:
            top_k: Number of experts to activate per token
            device: Device for created modules
        """
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_from_weights(
        self,
        expert_weights: List[torch.Tensor],
        expert_biases: Optional[List[torch.Tensor]],
        deviations: torch.Tensor,
    ) -> PACER_MoE_Layer:
        """
        Build an MoE layer from expert weight matrices.
        
        Args:
            expert_weights: List of weight tensors (d_out, d_in)
            expert_biases: Optional list of bias tensors (d_out,)
            deviations: Deviation tensors for routing (N, d_out, d_in)
            
        Returns:
            PACER_MoE_Layer instance
        """
        num_experts = len(expert_weights)
        d_out, d_in = expert_weights[0].shape
        
        # Create expert linear layers
        experts = []
        for i in range(num_experts):
            linear = nn.Linear(d_in, d_out, bias=(expert_biases is not None))
            linear.weight.data = expert_weights[i]
            if expert_biases is not None:
                linear.bias.data = expert_biases[i]
            experts.append(linear.to(self.device))
        
        # Create router
        router = ZeroShotRouter(
            expert_deviations=deviations.to(self.device),
            top_k=min(self.top_k, num_experts),
        )
        
        return PACER_MoE_Layer(experts, router)
    
    def build_from_models(
        self,
        models: List[nn.Module],
        layer_name: str,
        consensus_weight: torch.Tensor,
    ) -> PACER_MoE_Layer:
        """
        Build an MoE layer by extracting weights from multiple models.
        
        Args:
            models: List of source models
            layer_name: Name of the layer to extract
            consensus_weight: Consensus weight for computing deviations
            
        Returns:
            PACER_MoE_Layer instance
        """
        expert_weights = []
        expert_biases = []
        deviations = []
        
        for model in models:
            # Navigate to the layer
            parts = layer_name.split(".")
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            
            weight = module.weight.data
            expert_weights.append(weight)
            
            if module.bias is not None:
                expert_biases.append(module.bias.data)
            
            # Compute deviation from consensus
            deviation = weight.float() - consensus_weight.float()
            deviations.append(deviation)
        
        deviations_tensor = torch.stack(deviations)
        biases = expert_biases if expert_biases else None
        
        return self.build_from_weights(expert_weights, biases, deviations_tensor)
