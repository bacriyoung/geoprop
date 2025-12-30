
---

# GeoProp: Geometric-Propagated Pseudo-Label Generation

GeoProp is a multi-stage framework designed for high-fidelity pseudo-label generation in large-scale 3D scenes. By integrating decoupled feature learning with geometric post-processing, it generates structured and noise-resistant labels suitable for downstream weakly-supervised learning.

## Pipeline Overview

The framework processes 3D point clouds through a four-stage refinement pipeline:

1. **Stage 1: Semantic Initialization**: Uses `PointJAFAR` for decoupled feature learning, followed by **Test-Time Augmentation (TTA)** to consolidate semantic consistency.
2. **Stage 2: Geometric Gating**: Implements voxel-based geometric voting with a **Confidence-Aware Protection** mechanism to preserve high-frequency details (e.g., chair legs) while smoothing planar regions.
3. **Stage 3: Graph Refine**: A graph-cut based optimization that aligns semantic boundaries with geometric edges, protected by semantic confidence maps to prevent over-smoothing.
4. **Stage 4: Spatial Smoothing**: Final spatial consistency check using K-Nearest Neighbors (KNN).

## Project Structure

```text
geoprop/
├── config/             # YAML configurations (Global & Dataset-specific)
├── core/               # Training and Inference logic
│   └── modules/        # post-processing modules
├── data/               # Data loaders and seed generation
├── models/             # PointJAFAR model architecture
├── utils/              # Metrics, Visualization, and Logging
├── main.py             # Unified entry point
└── outputs/            # Time-stamped experiment results

```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-repo/geoprop.git
cd geoprop

```


2. **Environment Setup**:
```bash
# Recommendation: Use Mamba or Conda
mamba create -n geoprop python=3.9
mamba activate geoprop
pip install torch torchvision torchaudio
```



## Data Preparation

1. **Download S3DIS Dataset**:
Download the raw `.npy` files for S3DIS (Areas 1-6):
```bash
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y

```


2. **Configuration**:
Update the `root_dir` in `geoprop/config/s3dis/s3dis.yaml` to point to your data location.

## Usage

GeoProp is designed to run from the package root directory.

### Full Pipeline (Train + Generate)

To train the model on sparse labels and generate the full post-processed pseudo-labels:

```bash
python main.py --global_config config/global.yaml

```

### Configuration Details

You can toggle specific post-processing stages in `config/global.yaml`:

* `geometric_gating.confidence_threshold`: Adjusts the boundary between semantic and geometric priority.
* `inference.save_img`: Toggles visualization output.
* `inference.tta.rounds`: Sets the number of TTA iterations.

## Experiment Logging

Experiments are automatically organized by timestamp:
`geoprop/outputs/s3dis/YYYYMMDD_HHMMSS/`

* `logs/`: Detailed pipeline logs including per-class IoU analysis.
* `viz/`: Visualization of each refinement stage.
* `pseudo_labels/`: The final generated `.npy` files with refined labels.

## Citation

If you find this work helpful in your research, please consider citing our project.

---

