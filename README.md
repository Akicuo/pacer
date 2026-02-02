#  PacerKit

**PACER: Permutation-Aligned Consensus Expert Routing**

A unified framework for base-free, interference-aware model merging in Large Language Models and Vision Transformers.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

##  Key Features

- **No Base Model Required** - Synthesizes a Consensus Barycenter from input models
- **Interference-Aware** - Dynamically decides between merging and MoE upcycling per layer
- **Smart Routing** - Zero-shot router using Subspace Projection Affinity (no training needed)
- **Vision Support** - Native ViT support with Visual Token Merging (ToMe)
- **Minimal Parameter Growth** - Only upcycles high-conflict layers to MoE
- **Activation-Guided Retention** - Keep activated layers/experts from dense or MoE models via vLLM prompts

---

##  Installation

### Quick Install

```bash
git clone https://github.com/Akicuo/pacer.git
cd pacer
pip install -e .
```

### Manual Installation

```bash
pip install torch transformers safetensors accelerate
pip install -r requirements.txt
```

---

##  Quick Start

### Python API

```python
from pacerkit import PACERMerger

# can be None or {} to use default merging method
# merge_config = None
# merge_config = {}

# set to use activated experts or layers through vllm inference and p(ositive) prompts
merge_config = {
    "models": [
        {
            "hf_id": "fluently/FluentlyQwen3-Coder-4B-0909",
            "p_prompts": [
                "Write a Typescript function that requests a chat completion through the OpenAI Client",
                "Golang function that checks whether any foreign requests are being sent to third parties"
            ]
        },
        {
            "hf_id": "SamuelBang/AesCoder-4B",
            "p_prompts": [
                "Using HTML CSS JS and React - make a beautiful responsive landing page",
                "Using HTML CSS JS and Tailwind via cdn - Design a brutalist minimaist ecommerce webpage"
            ]
        }
    ]
}

# Initialize merger with models (everything is set in merge_config)
merger = PACERMerger(config=merge_config)

# Run PACER merge pipeline
merged_model = merger.merge(
    interference_threshold=0.35,
    top_k_experts=2,
    output_path="./merged_model"
)
```

### CLI

```bash
# Merge models using a config file
pacerkit merge --config configs/qwen_coder_merge.yaml

# Analyze interference between models
pacerkit analyze --models model1 model2 --output report.json
```

### Jupyter Notebook

See [`notebooks/pacer_quickstart.ipynb`](notebooks/pacer_quickstart.ipynb) for an interactive guide.

---

##  Configuration

PacerKit uses YAML configuration files:

```yaml
project_name: "qwen-coder-merge"

models:
  - "fluently/FluentlyQwen3-Coder-4B-0909"
  - "SamuelBang/AesCoder-4B"

output:
  path: "./merged_model"
  save_format: "safetensors"

pacer:
  interference_threshold: 0.35
  top_k_experts: 2
  dropout_rate: 0.1
  anchor_strategy: "first"
  enable_moe_upcycle: true
```

### Activation-Guided YAML Example

```yaml
project_name: "activation-guided-merge"

models:
  - hf_id: "fluently/FluentlyQwen3-Coder-4B-0909"
    p_prompts:
      - "Write a Typescript function that requests a chat completion through the OpenAI Client"
      - "Golang function that checks whether any foreign requests are being sent to third parties"
  - hf_id: "SamuelBang/AesCoder-4B"
    p_prompts:
      - "Using HTML CSS JS and React - make a beautiful responsive landing page"
      - "Using HTML CSS JS and Tailwind via cdn - Design a brutalist minimaist ecommerce webpage"

activation:
  enabled: true
  backend: "vllm"
  keep_top_layers: 10
  keep_top_experts_per_layer: 2
  min_activation_score: 0.0
  max_prompts: 8
  max_tokens: 1
  temperature: 0.0

output:
  path: "./merged_model"
  save_format: "safetensors"

pacer:
  interference_threshold: 0.35
  top_k_experts: 2
  dropout_rate: 0.1
  anchor_strategy: "first"
  enable_moe_upcycle: true
```

See [`configs/`](configs/) for more examples.

Activation-guided merging can also be configured via dict input (see Python API) or YAML with prompt lists and activation settings. The merge output folder automatically includes a `README.md` and `merge_config.json` describing the merge and activation retention.

---

##  How It Works

PACER operates in three phases:

### Phase 1: Geometric Alignment (Git Re-Basin)
Aligns permutation symmetries of N models into a shared geometric basin using weight matching and the Hungarian algorithm.

### Phase 2: Consensus Barycenter
Computes the Fréchet Mean of aligned models to create a synthetic "base model", then calculates deviation vectors.

### Phase 3: Interference-Aware Upcycling
- **Low interference layers** → DARE-TIES merge (0% parameter increase)
- **High interference layers** → MoE upcycling with zero-shot routing

---

##  Performance

| Metric | Dense Ensemble (4x) | Standard MoE | PACER |
|--------|---------------------|--------------|-------|
| **Total Params** | 400% | 400% | **~136%** |
| **Active Params** | 400% | 100% | **~100%** |
| **Interference** | None | Low | **None** |

---

## 📚 Documentation

- [Methodology](docs/methodology.md) - Full technical details
- [Configuration Reference](docs/configuration.md) - All config options
- [API Reference](docs/api.md) - Python API documentation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built on research from:
- Git Re-Basin (Ainsworth et al.)
- TIES-Merging (Yadav et al.)
- Token Merging (Bolya et al.)
- MergeME (Model Merging for MoEs)
- Claude Code Max for helping me through this Project <3
