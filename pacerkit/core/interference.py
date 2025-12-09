"""
Interference analysis for PACER.

This module computes the interference metric for each layer to determine
whether to merge (low interference) or upcycle to MoE (high interference).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


class InterferenceAnalyzer:
    """
    Analyze interference between model weights to guide merge decisions.
    
    The interference metric measures how much the deviation vectors
    conflict with each other. High interference means models disagree
    about the direction of change for a parameter.
    
    Interference Metric:
        I = 1 - ||Σ Δ_k||₂ / Σ ||Δ_k||₂
        
    - I ≈ 0: Low interference (deviations aligned, safe to merge)
    - I ≈ 1: High interference (deviations conflicting, needs MoE)
    """
    
    def __init__(
        self,
        threshold: float = 0.35,
        verbose: bool = True,
    ):
        """
        Initialize the interference analyzer.
        
        Args:
            threshold: Interference threshold for MoE upcycling decision
            verbose: Whether to show progress
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        
        self.threshold = threshold
        self.verbose = verbose
        
        # Stores per-layer interference scores
        self.interference_scores: Dict[str, float] = {}
        
        # Stores decisions per layer
        self.decisions: Dict[str, str] = {}
    
    def compute_interference(
        self,
        deviations: torch.Tensor,
    ) -> float:
        """
        Compute interference metric for a set of deviation vectors.
        
        Args:
            deviations: Tensor of shape (N, *param_shape) with deviation
                       vectors from N models
                       
        Returns:
            Interference score between 0 and 1
        """
        # Flatten each deviation to 1D
        N = deviations.shape[0]
        flat_devs = deviations.view(N, -1).float()
        
        # Compute norm of sum: ||Σ Δ_k||
        sum_of_devs = torch.sum(flat_devs, dim=0)
        norm_of_sum = torch.norm(sum_of_devs)
        
        # Compute sum of norms: Σ ||Δ_k||
        norms = torch.norm(flat_devs, dim=1)
        sum_of_norms = torch.sum(norms)
        
        # Avoid division by zero
        if sum_of_norms < 1e-8:
            return 0.0
        
        # Interference: 1 - (norm of sum / sum of norms)
        # When aligned: ratio ≈ 1, interference ≈ 0
        # When conflicting: ratio ≈ 0, interference ≈ 1
        interference = 1.0 - (norm_of_sum / sum_of_norms).item()
        
        return max(0.0, min(1.0, interference))
    
    def analyze_layer(
        self,
        layer_name: str,
        deviations: torch.Tensor,
    ) -> Tuple[float, str]:
        """
        Analyze a single layer and make a merge/upcycle decision.
        
        Args:
            layer_name: Name of the layer being analyzed
            deviations: Deviation vectors for this layer
            
        Returns:
            Tuple of (interference_score, decision)
            Decision is either "merge" or "upcycle_moe"
        """
        score = self.compute_interference(deviations)
        
        if score > self.threshold:
            decision = "upcycle_moe"
        else:
            decision = "merge"
        
        self.interference_scores[layer_name] = score
        self.decisions[layer_name] = decision
        
        return score, decision
    
    def analyze_all_layers(
        self,
        deviation_getter: callable,
        layer_names: List[str],
        filter_mlp_only: bool = True,
    ) -> Dict[str, Tuple[float, str]]:
        """
        Analyze all layers and compute interference scores.
        
        Args:
            deviation_getter: Function that takes layer_name and returns deviation tensor
            layer_names: List of layer names to analyze
            filter_mlp_only: If True, only analyze MLP layers for MoE conversion
            
        Returns:
            Dict mapping layer names to (score, decision) tuples
        """
        if self.verbose:
            print(f"🔍 Analyzing interference across {len(layer_names)} layers...")
            print(f"   Threshold: {self.threshold}")
        
        results = {}
        
        iterator = tqdm(layer_names, desc="Analyzing layers") if self.verbose else layer_names
        
        for layer_name in iterator:
            # Filter for MLP layers if requested
            if filter_mlp_only:
                if not any(mlp_key in layer_name.lower() for mlp_key in 
                          ["mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj"]):
                    continue
            
            try:
                deviations = deviation_getter(layer_name)
                score, decision = self.analyze_layer(layer_name, deviations)
                results[layer_name] = (score, decision)
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠ Skipping {layer_name}: {e}")
        
        if self.verbose:
            merge_count = sum(1 for _, d in results.values() if d == "merge")
            moe_count = sum(1 for _, d in results.values() if d == "upcycle_moe")
            print(f"✓ Analysis complete!")
            print(f"   Merge layers: {merge_count}")
            print(f"   MoE layers: {moe_count}")
        
        return results
    
    def cluster_experts(
        self,
        deviations: torch.Tensor,
        similarity_threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        Cluster similar deviation vectors to reduce number of experts.
        
        If multiple models have similar deviations for a layer, they can
        share an expert, reducing parameter count.
        
        Args:
            deviations: Deviation vectors of shape (N, *param_shape)
            similarity_threshold: Cosine similarity threshold for clustering
            
        Returns:
            Tuple of:
                - Clustered expert weights (M, *param_shape) where M <= N
                - List of lists showing which original models map to each cluster
        """
        N = deviations.shape[0]
        
        if N <= 1:
            return deviations, [[0]]
        
        # Flatten and normalize for cosine similarity
        flat = deviations.view(N, -1).float()
        flat_norm = torch.nn.functional.normalize(flat, dim=1)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        
        # Convert to distance for clustering
        distance_matrix = 1 - sim_matrix.cpu().numpy()
        
        # Perform agglomerative clustering
        # distance_threshold = 1 - similarity_threshold (since distance = 1 - similarity)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric="precomputed",
            linkage="average",
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Group models by cluster
        num_clusters = len(set(labels))
        clusters = [[] for _ in range(num_clusters)]
        for model_idx, cluster_idx in enumerate(labels):
            clusters[cluster_idx].append(model_idx)
        
        # Compute cluster centroids (averaged weights)
        clustered_weights = []
        for cluster_indices in clusters:
            cluster_devs = deviations[cluster_indices]
            centroid = torch.mean(cluster_devs, dim=0)
            clustered_weights.append(centroid)
        
        return torch.stack(clustered_weights), clusters
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of the interference analysis.
        
        Returns:
            Dict with analysis summary statistics
        """
        if not self.interference_scores:
            return {"status": "no analysis performed"}
        
        scores = list(self.interference_scores.values())
        
        return {
            "total_layers": len(scores),
            "merge_layers": sum(1 for d in self.decisions.values() if d == "merge"),
            "moe_layers": sum(1 for d in self.decisions.values() if d == "upcycle_moe"),
            "avg_interference": sum(scores) / len(scores),
            "max_interference": max(scores),
            "min_interference": min(scores),
            "threshold": self.threshold,
        }
    
    def get_high_interference_layers(
        self,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get layers with highest interference scores.
        
        Args:
            top_k: Number of layers to return (None for all)
            
        Returns:
            List of (layer_name, score) tuples sorted by score descending
        """
        sorted_layers = sorted(
            self.interference_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        if top_k is not None:
            sorted_layers = sorted_layers[:top_k]
        
        return sorted_layers
