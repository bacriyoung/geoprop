# GeoProp: Geometric-Semantic Synergy for Label-Efficient Point Cloud Segmentation

**GeoProp** is a weakly supervised pseudo-label generation framework designed for large-scale 3D point cloud segmentation.

Addressing the challenges of high-frequency noise and boundary ambiguity inherent in sparse annotation regimes, GeoProp introduces a **Geometric-Semantic Synergy** mechanism. By coupling decoupled feature learning with confidence-adaptive manifold regularization, the framework achieves high-fidelity transductive label propagation from extremely sparse seeds (Label Ratio < 0.1%) to dense scenes.

## Core Contributions

GeoProp establishes a complete solution for label-efficient learning through the following innovations:

* **Geometric-Semantic Synergy**: Resolves the trade-off between semantic consistency and geometric fidelity by using semantic predictions as a base and geometric priors as constraints.
* **Decoupled Feature Learning**: Integrates the `DecoupledPointJAFAR` module to explicitly separate semantic and geometric feature encoding, preventing feature entanglement under sparse supervision.
* **Confidence-Aware Geometric Gating**: Introduces a differentiable structural consistency constraint. By dynamically computing a confidence map, this module adaptively rectifies boundary ambiguities while protecting high-confidence fine-grained structures (e.g., chair legs).
* **Robust Transductive Inference**: Reformulates pseudo-label generation as an energy minimization problem on sparse graphs, achieving state-of-the-art quality.

## Architecture

The GeoProp framework operates in three streamlined phases:

1. **Training Phase**: The decoupled feature extractor (PointJAFAR) is trained using sparse seed annotations to learn robust semantic representations.
2. **Inference Phase**: Initial semantic probability distributions are generated. This phase incorporates **Test-Time Augmentation (TTA)** and a **Confidence Filter** to produce a robust semantic initialization.
3. **Post-processing Phase**: A cascade of geometric refinement modules:
* **Geometric Gating**: The core innovation that applies supervoxel-based geometric voting with confidence protection.
* **Graph Refinement**: Constructs a semantic-geometric affinity graph to optimize regional consistency via graph cuts.
* **Spatial Smoothing**: Final spatial consistency verification.



## Directory Structure

```text
geoprop/
├── config/             # Configuration system (Global & Dataset-specific)
├── core/               # Core computation engine
│   ├── modules/        # Refinement modules (Geometric Gating, Graph Refine, etc.)
│   ├── inferencer.py   # Full-scene inference pipeline
│   └── trainer.py      # Sparse supervision training logic
├── data/               # Data loading and processing
├── models/             # PointJAFAR network architecture
├── utils/              # Metrics and visualization tools
├── main.py             # Unified entry point
└── outputs/            # Experiment outputs (archived by timestamp)

```

## Installation

### Prerequisites

* Linux (Tested on Ubuntu 20.04)
* Python 3.8+
* PyTorch 1.10+ (CUDA 11.x recommended)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/geoprop.git
cd geoprop

# 2. Install dependencies
pip install -r requirements.txt

```

## Data Preparation

Currently supports the **S3DIS (Stanford Large-Scale 3D Indoor Spaces Dataset)**.

1. **Download Data**:
Download the pre-processed `.npy` files (containing XYZ, RGB, Label).
```bash
# Download via gdown
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y

```


2. **Configure Path**:
Update `root_dir` in `geoprop/config/s3dis/s3dis.yaml` to point to your data directory.

## Usage

GeoProp is designed for one-click execution, handling training, validation, inference, and optimization automatically.

### 1. Full Pipeline (Training + Generation)

This is the standard mode. The system performs sparse supervision training and then generates pseudo-labels for the full scene.

```bash
# Run from the project root directory
python main.py --global_config config/global.yaml

```

### 2. Inference Only

If you possess pre-trained weights and wish to test different post-processing parameters:

1. Modify `config/global.yaml`:
* Set `train: enable: false`
* Set `inference: checkpoint_path: "./outputs/s3dis/YOUR_TIMESTAMP/best_model.pth"`


2. Run:
```bash
python main.py

```



### 3. Key Configuration Parameters (`config/global.yaml`)

You can quickly verify ablation studies by adjusting the configuration:

```yaml
inference:
  # Geometric Gating Module
  geometric_gating:
    enabled: true
    gate_strength: 5.0
    confidence_threshold: 0.65  # Critical for protecting fine details

  # Graph Refinement Module
  graph_refine:
    enabled: true
    fine_voxel_n: 80            # Controls the granularity of the graph cut

```

## Performance

GeoProp demonstrates superior robustness on the S3DIS dataset, particularly in extremely label-scarce regimes. Experiment logs and generated pseudo-labels are automatically saved in:
`outputs/s3dis/YYYYMMDD_HHMMSS/`

## Citation

If you use GeoProp in your research, please cite our work:

```bibtex
@article{geoprop2025,
  title={GeoProp: Geometric-Semantic Synergy for Label-Efficient Point Cloud Segmentation},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:25xx.xxxxx},
  year={2025}
}

```

## Contact

For any questions, please contact us via issues or email.

---

### Acknowledgement

This project is built upon insights from weakly supervised learning and 3D geometric processing. We thank the open-source community for their contributions.