# 🚀 PacerKit

**PACER: Permutation-Aligned Consensus Expert Routing**

A unified framework for base-free, interference-aware model merging in Large Language Models and Vision Transformers.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## ✨ Key Features

- **🔓 No Base Model Required** - Synthesizes a Consensus Barycenter from input models
- **🎯 Interference-Aware** - Dynamically decides between merging and MoE upcycling per layer
- **🧠 Smart Routing** - Zero-shot router using Subspace Projection Affinity (no training needed)
- **👁️ Vision Support** - Native ViT support with Visual Token Merging (ToMe)
- **📦 Minimal Parameter Growth** - Only upcycles high-conflict layers to MoE

---

## 📦 Installation

### Quick Install

```bash
git clone https://github.com/yourusername/pacerkit.git
cd pacerkit
pip install -e .
```

### Manual Installation

```bash
pip install torch transformers safetensors accelerate
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Python API

```python
from pacerkit import PACERMerger

# Initialize merger with models
merger = PACERMerger([
    "fluently/FluentlyQwen3-Coder-4B-0909",
    "SamuelBang/AesCoder-4B"
])

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

## ⚙️ Configuration

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

See [`configs/`](configs/) for more examples.

---

## 🔬 How It Works

PACER operates in three phases:

### Phase 1: Geometric Alignment (Git Re-Basin)
Aligns permutation symmetries of N models into a shared geometric basin using weight matching and the Hungarian algorithm.

### Phase 2: Consensus Barycenter
Computes the Fréchet Mean of aligned models to create a synthetic "base model", then calculates deviation vectors.

### Phase 3: Interference-Aware Upcycling
- **Low interference layers** → DARE-TIES merge (0% parameter increase)
- **High interference layers** → MoE upcycling with zero-shot routing

---

## 📊 Performance

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
