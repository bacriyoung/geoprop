# GeoProp: Geometric-Semantic Synergy for Label-Efficient Point Cloud Segmentation

**GeoProp** proposes a transductive framework for high-fidelity 3D point cloud segmentation under **extreme label scarcity (label ratio < 0.1%)**.

By establishing a **synergistic coupling** between decoupled semantic representation learning and confidence-adaptive geometric priors, this framework effectively mitigates the inherent trade-off between semantic consistency and boundary precision in weakly supervised regimes.

## 1. Abstract

Semi-supervised learning on 3D point clouds often suffers from high-frequency noise and boundary ambiguities when supervision is sparse. We address this by reformulating the pseudo-label generation process as an energy minimization problem on a semantic-geometric affinity graph. **GeoProp** introduces a **Geometric Gating** mechanism that serves as a differentiable structural prior, adaptively rectifying semantic predictions while preserving fine-grained geometric details through a confidence-aware manifold regularization strategy.

## 2. Methodology

The framework operates in a coarse-to-fine manner, integrating three core components:

* **Decoupled Feature Learning**: A dual-stream architecture (`PointJAFAR`) that explicitly disentangles semantic feature encoding from geometric embedding, preventing feature collapse under sparse supervision.
* **Geometric-Semantic Synergy**: A novel post-processing paradigm where semantic probabilities act as the base signal, constrained by supervoxel-based geometric voting and graph-cut optimization.
* **Confidence-Aware Propagation**: A dynamic filtering mechanism that eliminates high-frequency noise while protecting high-confidence structural predictions (e.g., thin structures like chair legs) from over-smoothing.

## 3. Quick Start

### Installation

```bash
git clone https://github.com/yourusername/geoprop.git
cd geoprop
pip install -r requirements.txt

```

### Data Preparation (S3DIS)

Download the pre-processed S3DIS dataset and configure `root_dir` in `config/s3dis/s3dis.yaml`.

```bash
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y

```

### Inference & Evaluation

GeoProp supports two operation modes via `config/global.yaml`:

**1. Production Mode (Default)**

* Optimized for deployment efficiency.
* Outputs final pseudo-labels and logs detailed per-class IoU.
```bash
python main.py

```



**2. Ablation Mode**

* Designed for mechanism analysis.
* Saves intermediate results from specific modules (Geometric Gating, Graph Refine, etc.) to analyze the progressive refinement quality.
```bash
# Set 'ablation_mode: true' in config/global.yaml
python main.py

```



## 4. Citation

If you use GeoProp in your research, please cite our work.

---

*Disclaimer: This project is a research prototype designed for label-efficient 3D scene understanding.*