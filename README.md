# GeoProp: Geometric-Semantic Synergy for Label-Efficient Point Cloud Segmentation

**GeoProp** is a weakly supervised pseudo-label generation framework designed for large-scale 3D point cloud segmentation.

Addressing the challenges of high-frequency noise and boundary ambiguity inherent in sparse annotation regimes, GeoProp introduces a **Geometric-Semantic Synergy** mechanism. By coupling decoupled feature learning with confidence-adaptive manifold regularization, the framework achieves high-fidelity transductive label propagation from extremely sparse seeds (Label Ratio < 0.1%) to dense scenes.

## Core Contributions

GeoProp establishes a complete solution for label-efficient learning through the following innovations:

* **Geometric-Semantic Synergy**: Resolves the trade-off between semantic consistency and geometric fidelity by using semantic predictions as a base and geometric priors as constraints.
* **Decoupled Feature Learning**: Integrates the `DecoupledPointJAFAR` module to explicitly separate semantic and geometric feature encoding, preventing feature entanglement under sparse supervision.
* **Confidence-Aware Geometric Gating**: Introduces a differentiable structural consistency constraint. By dynamically computing a confidence map, this module adaptively rectifies boundary ambiguities while protecting high-confidence fine-grained structures.
* **Robust Transductive Inference**: Reformulates pseudo-label generation as an energy minimization problem on sparse graphs, achieving state-of-the-art quality.

## Architecture

The GeoProp framework operates in a streamlined Coarse-to-Fine pipeline:

1. **Training Phase**: The decoupled feature extractor is trained using sparse seed annotations to learn robust semantic representations.
2. **Inference Phase**:
* **Direct Inference**: Initial prediction from the trained model.
* **Test-Time Augmentation (TTA)**: Multi-view voting to stabilize probability distributions.
* **Confidence Filter**: Eliminates high-frequency noise based on prediction confidence.


3. **Refinement Phase**:
* **Geometric Gating**: Applies supervoxel-based geometric voting with confidence protection to rectify structural errors.
* **Graph Refinement**: Constructs a semantic-geometric affinity graph to optimize regional consistency via graph cuts.
* **Spatial Smoothing**: Final spatial consistency verification using KNN.



## Directory Structure

```text
geoprop/
├── config/             # Configuration system (Global & Dataset-specific)
├── core/               # Core computation engine
│   ├── modules/        # Refinement modules (Geometric Gating, Graph Refine, etc.)
│   ├── inferencer.py   # Full-scene inference pipeline with dual modes
│   └── trainer.py      # Sparse supervision training logic
├── data/               # Data loading and processing
├── models/             # Decoupled PointJAFAR architecture
├── utils/              # Metrics, Logger, and Visualization tools
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
git clone https://github.com/bacriyoung/geoprop.git
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

GeoProp supports two distinct operation modes controlled by the `ablation_mode` flag in `global.yaml`.

### 1. Production Mode (Default)

Optimized for speed and deployment. It processes the full pipeline but only saves the final result.

* **Behavior**: Only the final stage (Spatial Smooth) is saved.
* **Logging**: detailed **Per-Class IoU** metrics are printed for every room to monitor specific category performance.
* **Command**:
```bash
python main.py --global_config config/global.yaml

```



### 2. Ablation Mode

Designed for research and debugging. It allows saving intermediate results from specific modules.

* **Behavior**: Intermediate pseudo-labels are saved into subdirectories (e.g., `pseudo_labels/Geometric_Gating/`) based on the `save_output` flag of each module.
* **Logging**: Prints stage-wise mIoU comparisons to track performance gains across modules.
* **Configuration**: Set `ablation_mode: true` in `global.yaml`.

### Inference Only

To skip training and run inference using a pre-trained model:

1. Modify `config/global.yaml`:
* Set `train: enable: false`
* Set `inference: checkpoint_path: "./outputs/s3dis/YOUR_TIMESTAMP/best_model.pth"`


2. Run `python main.py`

## Configuration

The behavior of GeoProp is fully controlled via `config/global.yaml`.

```yaml
inference:
  # Mode Selection
  ablation_mode: false   # false = Production Mode; true = Ablation Mode

  # Module Configuration
  geometric_gating:
    enabled: true
    gate_strength: 5.0
    confidence_threshold: 0.65  # Protects fine details
    save_output: true           # Only effective in Ablation Mode

  graph_refine:
    enabled: true
    fine_voxel_n: 80
    save_output: true

```

## Output & Logging

All results are automatically organized by timestamp to prevent overwriting: `outputs/s3dis/YYYYMMDD_HHMMSS/`

* **Logs**: `logs/pipeline_*.log` contains key configuration summaries and detailed evaluation metrics.
* **Pseudo Labels**:
* **Production Mode**: Saved directly in `pseudo_labels/`.
* **Ablation Mode**: Organized in subfolders like `pseudo_labels/Geometric_Gating/`.


* **Visualizations**: Saved in `viz/` (if enabled).

## Citation

If you use GeoProp in your research, please cite our work.

## Contact

For any questions, please contact us via issues or email.

---

### Acknowledgement

This project is built upon insights from weakly supervised learning and 3D geometric processing. We thank the open-source community for their contributions.