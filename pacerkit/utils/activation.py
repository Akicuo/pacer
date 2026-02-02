"""
Activation collection utilities for PACER.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from pacerkit.config import ActivationSettings, ModelPromptConfig


_EXPERT_REGEX = re.compile(r"(?:experts|expert)\.(\d+)")


def collect_activations(
    model_prompts: List[ModelPromptConfig],
    settings: ActivationSettings,
    trust_remote_code: bool = True,
) -> Dict[str, Any]:
    """
    Collect activation stats for each model using the configured backend.
    
    Returns:
        Dict containing activation stats and settings metadata.
    """
    if not model_prompts:
        return {
            "backend": settings.backend,
            "settings": asdict(settings),
            "models": {},
        }
    
    if settings.backend == "transformers":
        model_stats = _collect_transformers(model_prompts, settings, trust_remote_code)
    else:
        model_stats = _collect_vllm(model_prompts, settings, trust_remote_code)
    
    return {
        "backend": settings.backend,
        "settings": asdict(settings),
        "models": model_stats,
    }


def _collect_vllm(
    model_prompts: List[ModelPromptConfig],
    settings: ActivationSettings,
    trust_remote_code: bool,
) -> Dict[str, Dict[str, Any]]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for activation collection. "
            "Install with: pip install vllm"
        ) from exc
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for model_cfg in model_prompts:
        prompts = _truncate_prompts(model_cfg.p_prompts, settings.max_prompts)
        if not prompts:
            results[model_cfg.hf_id] = {"layers": {}, "experts": {}}
            continue
        
        llm = LLM(model=model_cfg.hf_id, trust_remote_code=trust_remote_code)
        model = _extract_vllm_model(llm)
        
        stats, handles = _register_activation_hooks(model)
        try:
            sampling = SamplingParams(
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )
            llm.generate(prompts, sampling_params=sampling)
        finally:
            for handle in handles:
                handle.remove()
        
        results[model_cfg.hf_id] = _summarize_stats(stats)
    
    return results


def _collect_transformers(
    model_prompts: List[ModelPromptConfig],
    settings: ActivationSettings,
    trust_remote_code: bool,
) -> Dict[str, Dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for model_cfg in model_prompts:
        prompts = _truncate_prompts(model_cfg.p_prompts, settings.max_prompts)
        if not prompts:
            results[model_cfg.hf_id] = {"layers": {}, "experts": {}}
            continue
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_id,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        stats, handles = _register_activation_hooks(model)
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                _ = model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        
        results[model_cfg.hf_id] = _summarize_stats(stats)
    
    return results


def _truncate_prompts(prompts: List[str], max_prompts: int | None) -> List[str]:
    if max_prompts is None:
        return list(prompts)
    return list(prompts)[: max_prompts]


def _extract_vllm_model(llm: Any) -> nn.Module:
    candidates = [
        ("llm_engine", "model_executor", "driver_worker", "model"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "worker", "model"),
        ("llm_engine", "model_executor", "worker", "model_runner", "model"),
    ]
    
    for path in candidates:
        obj = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.Module):
            return obj
    
    raise RuntimeError(
        "Unable to access the underlying vLLM model for activation hooks. "
        "Try setting activation.backend='transformers'."
    )


def _register_activation_hooks(
    model: nn.Module,
) -> Tuple[Dict[str, Dict[str, float]], List[Any]]:
    stats: Dict[str, Dict[str, float]] = {}
    handles: List[Any] = []
    
    for name, module in model.named_modules():
        if not _should_track_module(module):
            continue
        stats[name] = {"count": 0.0, "score": 0.0}
        
        def _hook(mod, inputs, output, module_name=name):
            stats[module_name]["count"] += 1.0
            value = _extract_activation_score(output)
            stats[module_name]["score"] += value
        
        handles.append(module.register_forward_hook(_hook))
    
    return stats, handles


def _should_track_module(module: nn.Module) -> bool:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return True
    # Track MoE expert modules if they have parameters
    return any(p.requires_grad for p in module.parameters(recurse=False))


def _extract_activation_score(output: Any) -> float:
    if torch.is_tensor(output):
        return float(output.detach().abs().mean().item())
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if torch.is_tensor(item):
                return float(item.detach().abs().mean().item())
    return 0.0


def _summarize_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    layers: Dict[str, float] = {}
    experts: Dict[str, Dict[str, float]] = {}
    
    for name, entry in stats.items():
        score = entry["score"] if entry["score"] > 0 else entry["count"]
        layers[name] = score
        
        match = _EXPERT_REGEX.search(name)
        if match:
            expert_id = match.group(1)
            prefix = name[: match.start()].rstrip(".")
            experts.setdefault(prefix, {})[expert_id] = score
    
    return {"layers": layers, "experts": experts}
