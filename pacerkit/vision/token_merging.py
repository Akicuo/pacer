"""
Visual Token Merging (ToMe) for Vision Transformers.

This module implements Token Merging to reduce redundant visual patches
before MoE routing, improving efficiency for Vision Transformer merging.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToMEConfig:
    """Configuration for Token Merging."""
    
    # Fraction of tokens to merge (0.0 to 1.0)
    merge_ratio: float = 0.3
    
    # Whether to merge before MoE routing
    merge_before_routing: bool = True
    
    # Minimum tokens to keep after merging
    min_tokens: int = 4
    
    # Similarity metric for matching
    similarity: str = "cosine"  # or "dot"
    
    # Whether to use bipartite matching
    bipartite: bool = True


class TokenMerger(nn.Module):
    """
    Token Merging module for Vision Transformers.
    
    Reduces the number of visual tokens by merging similar patches,
    reducing computational cost while preserving important information.
    """
    
    def __init__(self, config: Optional[ToMEConfig] = None):
        """
        Initialize Token Merger.
        
        Args:
            config: ToME configuration (uses defaults if None)
        """
        super().__init__()
        self.config = config or ToMEConfig()
    
    def compute_similarity(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between tokens.
        
        Args:
            tokens: Tensor of shape (batch, num_tokens, dim)
            
        Returns:
            Similarity matrix of shape (batch, num_tokens, num_tokens)
        """
        if self.config.similarity == "cosine":
            tokens_norm = F.normalize(tokens, dim=-1)
            return torch.bmm(tokens_norm, tokens_norm.transpose(-1, -2))
        else:
            # Dot product
            return torch.bmm(tokens, tokens.transpose(-1, -2))
    
    def bipartite_soft_matching(
        self,
        tokens: torch.Tensor,
        r: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform bipartite soft matching to select tokens to merge.
        
        Args:
            tokens: Tensor of shape (batch, num_tokens, dim)
            r: Number of tokens to remove
            
        Returns:
            Tuple of:
                - merged tokens
                - unmerged tokens
                - merge indices for potential unmerging
        """
        batch, num_tokens, dim = tokens.shape
        
        if r <= 0 or r >= num_tokens:
            return tokens, torch.empty(0), torch.arange(num_tokens)
        
        # Split tokens into two sets (source and destination)
        # Source tokens will be merged into destination tokens
        split = num_tokens // 2
        src = tokens[:, :split]  # First half
        dst = tokens[:, split:]  # Second half
        
        # Compute similarity between src and dst
        src_norm = F.normalize(src, dim=-1)
        dst_norm = F.normalize(dst, dim=-1)
        similarity = torch.bmm(src_norm, dst_norm.transpose(-1, -2))  # (B, split, num-split)
        
        # Find best matches for each source token
        scores, indices = similarity.max(dim=-1)  # (B, split)
        
        # Select top-r source tokens to merge (based on highest similarity)
        _, merge_idx = scores.topk(r, dim=-1, largest=True)  # (B, r)
        
        # Create mask for tokens to keep
        keep_mask = torch.ones(batch, split, dtype=torch.bool, device=tokens.device)
        keep_mask.scatter_(1, merge_idx, False)
        
        # Get tokens to keep from source
        kept_src = src[keep_mask.unsqueeze(-1).expand_as(src)].view(batch, -1, dim)
        
        # Merge selected source tokens into their matched destinations
        for b in range(batch):
            for i, src_idx in enumerate(merge_idx[b]):
                dst_idx = indices[b, src_idx]
                # Weighted average merge
                dst[b, dst_idx] = (dst[b, dst_idx] + src[b, src_idx]) / 2
        
        # Concatenate kept source tokens with (modified) destination tokens
        merged = torch.cat([kept_src, dst], dim=1)
        
        return merged, merge_idx, indices
    
    def forward(
        self,
        tokens: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Merge tokens to reduce sequence length.
        
        Args:
            tokens: Tensor of shape (batch, num_tokens, dim)
            cls_token: Optional CLS token to preserve (batch, 1, dim)
            
        Returns:
            Tuple of:
                - Merged tokens (batch, new_num_tokens, dim)
                - Merge info dict for potential unmerging
        """
        batch, num_tokens, dim = tokens.shape
        
        # Compute number of tokens to remove
        r = int(num_tokens * self.config.merge_ratio)
        r = min(r, num_tokens - self.config.min_tokens)
        
        if r <= 0:
            return tokens, None
        
        if self.config.bipartite:
            merged, merge_idx, match_idx = self.bipartite_soft_matching(tokens, r)
        else:
            # Simple greedy merging (fallback)
            merged = self._greedy_merge(tokens, r)
            merge_idx = None
            match_idx = None
        
        # Prepend CLS token if provided
        if cls_token is not None:
            merged = torch.cat([cls_token, merged], dim=1)
        
        merge_info = {
            "original_length": num_tokens,
            "merged_length": merged.shape[1],
            "merge_indices": merge_idx,
            "match_indices": match_idx,
        }
        
        return merged, merge_info
    
    def _greedy_merge(
        self,
        tokens: torch.Tensor,
        r: int,
    ) -> torch.Tensor:
        """
        Simple greedy token merging.
        
        Args:
            tokens: Tensor of shape (batch, num_tokens, dim)
            r: Number of tokens to remove
            
        Returns:
            Merged tokens
        """
        batch, num_tokens, dim = tokens.shape
        
        # Compute full similarity matrix
        sim = self.compute_similarity(tokens)
        
        # Mask diagonal
        mask = torch.eye(num_tokens, device=tokens.device).bool()
        sim.masked_fill_(mask.unsqueeze(0), float('-inf'))
        
        # Iteratively merge most similar pairs
        current_tokens = tokens.clone()
        keep_mask = torch.ones(batch, num_tokens, dtype=torch.bool, device=tokens.device)
        
        for _ in range(r):
            # Find most similar pair
            flat_sim = sim.view(batch, -1)
            max_idx = flat_sim.argmax(dim=-1)
            
            # Convert to row, col indices
            row = max_idx // sim.shape[-1]
            col = max_idx % sim.shape[-1]
            
            # Merge: average the pair, mark one as removed
            for b in range(batch):
                i, j = row[b].item(), col[b].item()
                if keep_mask[b, i] and keep_mask[b, j]:
                    current_tokens[b, i] = (current_tokens[b, i] + current_tokens[b, j]) / 2
                    keep_mask[b, j] = False
                    sim[b, :, j] = float('-inf')
                    sim[b, j, :] = float('-inf')
        
        # Gather kept tokens
        result = []
        for b in range(batch):
            kept = current_tokens[b, keep_mask[b]]
            result.append(kept)
        
        # Pad to same length (if needed)
        max_len = max(r.shape[0] for r in result)
        padded = torch.zeros(batch, max_len, dim, device=tokens.device, dtype=tokens.dtype)
        for b, r in enumerate(result):
            padded[b, :r.shape[0]] = r
        
        return padded
    
    def unmerge(
        self,
        merged_tokens: torch.Tensor,
        merge_info: dict,
        original_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Unmerge tokens back to original shape (for dense prediction).
        
        Args:
            merged_tokens: Merged token tensor
            merge_info: Info dict from forward pass
            original_shape: Original (batch, num_tokens, dim) shape
            
        Returns:
            Tokens restored to original shape
        """
        batch, orig_len, dim = original_shape
        
        if merge_info is None:
            return merged_tokens
        
        # Simple broadcast: duplicate merged tokens
        # In practice, this would use merge_info for proper mapping
        output = torch.zeros(original_shape, device=merged_tokens.device, dtype=merged_tokens.dtype)
        
        # Broadcast merged tokens to fill original shape
        merged_len = merged_tokens.shape[1]
        repeat_factor = (orig_len + merged_len - 1) // merged_len
        
        expanded = merged_tokens.repeat(1, repeat_factor, 1)[:, :orig_len, :]
        
        return expanded
