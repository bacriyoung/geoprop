# GeoProp: Geometric-Semantic Synergy for Label-Efficient Point Cloud Segmentation

**GeoProp** is a weakly supervised framework designed to bridge the gap between sparse annotation and dense semantic segmentation in large-scale 3D point clouds.

By establishing a synergistic coupling between decoupled semantic representation and geometric priors, GeoProp mitigates the inherent boundary ambiguity and high-frequency noise observed in label-scarce regimes (label ratio < 0.1%). The framework reformulates pseudo-label generation as a confidence-adaptive manifold regularization problem, achieving state-of-the-art transductive inference performance on the S3DIS dataset.

## Core Methodology

GeoProp operates on a **Coarse-to-Fine** paradigm, integrating three key theoretical contributions:

* **Geometric-Semantic Synergy**: A novel mechanism that resolves the trade-off between semantic consistency and geometric fidelity, utilizing semantic predictions as a base and geometric priors as structural constraints.
* **Decoupled Feature Learning**: An explicit separation of semantic and geometric feature encoding to prevent feature entanglement under sparse supervision.
* **Confidence-Adaptive Regularization**: A differentiable Geometric Gating module that dynamically rectifies structural errors based on local confidence maps, followed by a graph-cut optimization for regional consistency.

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/geoprop.git
cd geoprop
pip install -r requirements.txt

```

### 2. Data Preparation (S3DIS)

Download the pre-processed data and configure the `root_dir` in `config/s3dis/s3dis.yaml`.

```bash
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y

```

### 3. Usage

GeoProp provides a unified entry point for both training and inference, controlled by `config/global.yaml`.

**Production Mode (Default)**
Trains the model on sparse seeds and generates high-fidelity pseudo-labels for the entire dataset.

```bash
python main.py

```

**Ablation Mode**
To analyze the contribution of specific modules (e.g., Geometric Gating), enable `ablation_mode: true` in the configuration. This will serialize intermediate results for detailed inspection.

## Configuration

All hyper-parameters and module switches are centralized in `config/global.yaml`.

* **`label_ratio`**: Controls the sparsity of supervision (Default: `0.001`).
* **`geometric_gating`**: Controls the strength of geometric structural constraints.
* **`ablation_mode`**: Toggles between efficient production deployment and detailed experimental logging.

## Citation

If you find GeoProp useful for your research, please cite our work.

---

*This project is released under the MIT License.*