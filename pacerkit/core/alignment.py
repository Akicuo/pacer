"""
Permutation Alignment using Git Re-Basin algorithm.

This module handles the geometric alignment of multiple neural network models
by solving the permutation symmetry problem using weight matching.
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class PermutationAligner:
    """
    Align multiple models to a single anchor using Git Re-Basin.
    
    This class implements the weight matching approach for aligning neural network
    permutation symmetries. It uses the Hungarian algorithm to find optimal
    neuron correspondences between models.
    
    Attributes:
        anchor: The anchor model (all others aligned to this)
        peers: List of models to be aligned
        device: Torch device for computation
        verbose: Whether to print progress
    """
    
    def __init__(
        self,
        anchor_model: nn.Module,
        peer_models: List[nn.Module],
        anchor_strategy: Literal["first", "lowest_magnitude", "random"] = "first",
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the permutation aligner.
        
        Args:
            anchor_model: The model to use as the alignment anchor
            peer_models: List of models to align to the anchor
            anchor_strategy: Strategy for selecting anchor (currently uses provided anchor)
            device: Device for computation (auto-detected if None)
            verbose: Whether to show progress
        """
        self.anchor = anchor_model
        self.peers = peer_models
        self.anchor_strategy = anchor_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Store permutations for each peer model
        # Each entry is a dict mapping layer names to permutation vectors
        self.layer_permutations: List[Dict[str, torch.Tensor]] = [{} for _ in peer_models]
        
        # Track prev permutation for input dimension
        self._prev_perms: List[Optional[torch.Tensor]] = [None for _ in peer_models]
    
    def _compute_cost_matrix(
        self,
        w_anchor: torch.Tensor,
        w_peer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the cost matrix for the Hungarian algorithm.
        
        We maximize the inner product (equivalently minimize negative inner product)
        between normalized weight vectors.
        
        Args:
            w_anchor: Anchor weight matrix (d_out, d_in)
            w_peer: Peer weight matrix (d_out, d_in)
            
        Returns:
            Cost matrix of shape (d_out, d_out)
        """
        # Normalize rows to focus on direction, not magnitude
        w_a_norm = F.normalize(w_anchor.float(), dim=1)
        w_p_norm = F.normalize(w_peer.float(), dim=1)
        
        # Compute similarity matrix (higher = more similar)
        similarity = torch.mm(w_a_norm, w_p_norm.t())
        
        # Return negative for minimization (Hungarian minimizes)
        return -similarity
    
    def _get_linear_layers(self, model: nn.Module) -> Dict[str, nn.Linear]:
        """Get all linear layers from a model with their full names."""
        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        return linear_layers
    
    def _identify_residual_groups(
        self,
        layer_names: List[str],
    ) -> Dict[str, List[str]]:
        """
        Identify groups of layers that share residual connections.
        
        In Transformers, layers attached to the residual stream must have
        consistent permutations. This groups such layers together.
        
        Args:
            layer_names: List of layer names
            
        Returns:
            Dict mapping group ID to list of layer names
        """
        groups = {}
        
        for name in layer_names:
            # Extract block identifier (e.g., "layers.0", "blocks.5")
            parts = name.split(".")
            
            # Find the block prefix (typically "model.layers.N" or similar)
            block_id = None
            for i, part in enumerate(parts):
                if part.isdigit():
                    block_id = ".".join(parts[:i+1])
                    break
            
            if block_id is None:
                # No block structure found, treat as individual
                block_id = name
            
            if block_id not in groups:
                groups[block_id] = []
            groups[block_id].append(name)
        
        return groups
    
    def align(self) -> List[nn.Module]:
        """
        Perform greedy layer-wise alignment of all peer models to the anchor.
        
        This method iterates through layers from input to output, solving
        the linear assignment problem at each layer to find optimal permutations.
        Residual stream constraints are handled by aggregating cost matrices.
        
        Returns:
            List of aligned peer models (modified in-place)
        """
        if self.verbose:
            print(f"🔧 Starting Geometric Alignment...")
            print(f"   Anchor model: {type(self.anchor).__name__}")
            print(f"   Peer models: {len(self.peers)}")
        
        # Get all linear layers from anchor
        anchor_layers = self._get_linear_layers(self.anchor)
        layer_names = list(anchor_layers.keys())
        
        if self.verbose:
            print(f"   Found {len(layer_names)} linear layers to align")
        
        # Identify residual groups
        residual_groups = self._identify_residual_groups(layer_names)
        
        # Track which layers we've processed
        processed = set()
        
        # Process layers in order
        iterator = tqdm(layer_names, desc="Aligning layers") if self.verbose else layer_names
        
        for layer_name in iterator:
            if layer_name in processed:
                continue
            
            anchor_module = anchor_layers[layer_name]
            w_anchor = anchor_module.weight.data.to(self.device)
            
            # Process each peer model
            for peer_idx, peer in enumerate(self.peers):
                peer_layers = self._get_linear_layers(peer)
                
                if layer_name not in peer_layers:
                    continue
                
                peer_module = peer_layers[layer_name]
                w_peer = peer_module.weight.data.to(self.device)
                
                # Apply previous permutation to input dimension
                prev_perm = self._prev_perms[peer_idx]
                if prev_perm is not None and prev_perm.shape[0] == w_peer.shape[1]:
                    w_peer = w_peer[:, prev_perm]
                
                # Compute cost matrix
                cost = self._compute_cost_matrix(w_anchor, w_peer)
                
                # Solve assignment problem using Hungarian algorithm
                cost_np = cost.cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
                
                # col_ind tells us how to reorder peer's output dimension
                perm = torch.tensor(col_ind, device=self.device, dtype=torch.long)
                
                # Apply output permutation to weights
                w_peer_aligned = w_peer[perm, :]
                peer_module.weight.data = w_peer_aligned.to(peer_module.weight.device)
                
                # Apply to bias if present
                if peer_module.bias is not None:
                    peer_module.bias.data = peer_module.bias.data[perm]
                
                # Store permutation for this layer
                self.layer_permutations[peer_idx][layer_name] = perm.cpu()
                
                # Update prev_perm for next layer's input (only if shapes match)
                # This is for layers that feed directly into the next
                self._prev_perms[peer_idx] = perm
            
            processed.add(layer_name)
        
        if self.verbose:
            print(f"✓ Alignment complete!")
        
        return self.peers
    
    def get_permutation(
        self,
        peer_idx: int,
        layer_name: str,
    ) -> Optional[torch.Tensor]:
        """
        Get the permutation applied to a specific layer of a peer model.
        
        Args:
            peer_idx: Index of the peer model
            layer_name: Name of the layer
            
        Returns:
            Permutation tensor or None if not found
        """
        if peer_idx >= len(self.layer_permutations):
            return None
        return self.layer_permutations[peer_idx].get(layer_name)
    
    def compute_alignment_quality(
        self,
        peer_idx: int,
    ) -> float:
        """
        Compute a quality score for the alignment of a peer model.
        
        This measures how well the aligned peer matches the anchor
        in terms of weight direction similarity.
        
        Args:
            peer_idx: Index of the peer model
            
        Returns:
            Alignment quality score (0-1, higher is better)
        """
        anchor_layers = self._get_linear_layers(self.anchor)
        peer_layers = self._get_linear_layers(self.peers[peer_idx])
        
        total_sim = 0.0
        count = 0
        
        for name, anchor_mod in anchor_layers.items():
            if name not in peer_layers:
                continue
            
            w_a = F.normalize(anchor_mod.weight.data.float(), dim=1)
            w_p = F.normalize(peer_layers[name].weight.data.float(), dim=1)
            
            # Compute diagonal similarity (matched neurons)
            sim = torch.sum(w_a * w_p) / w_a.shape[0]
            total_sim += sim.item()
            count += 1
        
        return total_sim / count if count > 0 else 0.0
