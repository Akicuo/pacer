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
        num_workers: int = 4,
        fast_mode: bool = True,
    ):
        """
        Initialize the permutation aligner.
        
        Args:
            anchor_model: The model to use as the alignment anchor
            peer_models: List of models to align to the anchor
            anchor_strategy: Strategy for selecting anchor (currently uses provided anchor)
            device: Device for computation (None = respect model's device placement)
            verbose: Whether to show progress
            num_workers: Number of parallel workers for peer alignment (default: 4)
            fast_mode: Use approximations for large layers (default: True)
        """
        self.anchor = anchor_model
        self.peers = peer_models
        self.anchor_strategy = anchor_strategy
        # If device is None, we'll use weights where they are (multi-GPU friendly)
        # If specified, we'll move to that device (backward compatibility)
        self.force_device = device
        self.verbose = verbose
        self.num_workers = num_workers
        self.fast_mode = fast_mode
        
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
            w_anchor: Anchor weight matrix (d_out, d_in) - can be on any device
            w_peer: Peer weight matrix (d_out, d_in) - can be on any device
            
        Returns:
            Cost matrix of shape (d_out, d_out) on CPU (for Hungarian algorithm)
        """
        # Check size to decide strategy
        d_out, d_in = w_anchor.shape
        matrix_elements = d_out * d_out
        
        # Determine device to use
        if self.force_device:
            # User specified a device - use it (backward compatibility)
            device = self.force_device
            use_cpu = device == "cpu"
        else:
            # Multi-GPU mode: use weight's current device if small enough, else CPU
            # Heuristic: if result matrix > 2GB, use CPU
            # 2GB = 5e8 floats
            use_cpu = matrix_elements > 5e8
            device = w_anchor.device if not use_cpu else torch.device("cpu")
        
        # Move to target device if needed
        if use_cpu or str(w_anchor.device) != str(device):
            w_a = w_anchor.float().to(device)
            w_p = w_peer.float().to(device)
        else:
            w_a = w_anchor.float()
            w_p = w_peer.float()
            
        # Normalize rows
        w_a_norm = F.normalize(w_a, dim=1)
        w_p_norm = F.normalize(w_p, dim=1)
        
        # Compute similarity matrix
        if use_cpu and matrix_elements > 2e9: # >8GB result
            # Chunked computation on CPU to save memory
            similarity = torch.empty((d_out, d_out), dtype=torch.float32, device=device)
            chunk_size = 1024
            for i in range(0, d_out, chunk_size):
                end = min(i + chunk_size, d_out)
                similarity[i:end] = torch.mm(w_a_norm[i:end], w_p_norm.t())
        else:
            # Standard computation
            similarity = torch.mm(w_a_norm, w_p_norm.t())
        
        # Return on CPU for Hungarian algorithm (scipy needs numpy)
        return -similarity.cpu()
    
    def _solve_assignment(
        self,
        cost: torch.Tensor,
        fast_mode: bool = None,
    ) -> torch.Tensor:
        """
        Solve the linear assignment problem.
        
        Args:
            cost: Cost matrix (n, n) on CPU
            fast_mode: Use greedy approximation for large matrices (default: self.fast_mode)
            
        Returns:
            Permutation tensor (column indices)
        """
        fast_mode = fast_mode if fast_mode is not None else self.fast_mode
        n = cost.shape[0]
        
        # Use greedy approximation for large layers in fast mode
        if fast_mode and n > 512:
            # Greedy matching: for each row, pick the best column
            # Not optimal but O(n²) instead of O(n³)
            # And often good enough for weight alignment
            perm = torch.zeros(n, dtype=torch.long)
            cost_np = cost.numpy()
            
            # Track which columns are taken
            available = set(range(n))
            
            # Sort rows by their minimum cost (process hardest first)
            row_mins = cost_np.min(axis=1)
            row_order = row_mins.argsort()
            
            for row in row_order:
                # Find best available column for this row
                best_col = None
                best_cost = float('inf')
                for col in available:
                    if cost_np[row, col] < best_cost:
                        best_cost = cost_np[row, col]
                        best_col = col
                
                perm[row] = best_col
                available.remove(best_col)
            
            return perm
        else:
            # Use exact Hungarian algorithm for small/medium layers
            cost_np = cost.numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            return torch.tensor(col_ind, dtype=torch.long)
    
    def _get_linear_layers(self, model: nn.Module) -> Dict[str, nn.Linear]:
        """Get all alignable linear layers (excluding heads)."""
        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Exclude lm_head and other final projection layers
                # Permuting vocab dimension destroys token semantics
                if "lm_head" in name or "embed" in name or "score" in name:
                    continue
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
            # Respect model's device placement (multi-GPU friendly)
            w_anchor = anchor_module.weight.data
            
            # Skip very small layers (< 32 neurons) - not worth aligning
            if w_anchor.shape[0] < 32:
                processed.add(layer_name)
                continue
            
            # Define processing function for a single peer
            def process_peer(peer_idx):
                peer = self.peers[peer_idx]
                peer_layers = self._get_linear_layers(peer)
                
                if layer_name not in peer_layers:
                    return None
                
                peer_module = peer_layers[layer_name]
                # Keep weights on their original device
                w_peer = peer_module.weight.data
                
                # Apply previous permutation to input dimension
                prev_perm = self._prev_perms[peer_idx]
                if prev_perm is not None and prev_perm.shape[0] == w_peer.shape[1]:
                    # Move perm to same device as weights
                    prev_perm_device = prev_perm.to(w_peer.device)
                    w_peer = w_peer[:, prev_perm_device]
                
                # Compute cost matrix (handles device automatically)
                cost = self._compute_cost_matrix(w_anchor, w_peer)
                
                # Solve assignment problem (fast mode for large layers)
                col_ind = self._solve_assignment(cost)
                
                # Keep perm on same device as the weights  
                perm = col_ind.to(w_peer.device)
                
                # Apply output permutation to weights
                w_peer_aligned = w_peer[perm, :]
                peer_module.weight.data = w_peer_aligned
                
                # Apply to bias if present
                if peer_module.bias is not None:
                    bias_device = peer_module.bias.device
                    perm_bias = perm.to(bias_device)
                    peer_module.bias.data = peer_module.bias.data[perm_bias]
                
                return perm.cpu()  # Return on CPU for storage
            
            # Process peers in parallel if multiple workers
            if self.num_workers > 1 and len(self.peers) > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    peer_indices = list(range(len(self.peers)))
                    results = list(executor.map(process_peer, peer_indices))
                    
                    # Store results
                    for peer_idx, perm in enumerate(results):
                        if perm is not None:
                            self.layer_permutations[peer_idx][layer_name] = perm
                            self._prev_perms[peer_idx] = perm
            else:
                # Sequential processing (single worker or single peer)
                for peer_idx in range(len(self.peers)):
                    perm = process_peer(peer_idx)
                    if perm is not None:
                        self.layer_permutations[peer_idx][layer_name] = perm
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
