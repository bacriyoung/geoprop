# AFA-Net

Official implementation of **Geometry-Constrained Asymmetric Feature Aggregation for Label-Efficient 3D Point Cloud Semantic Segmentation**.

AFA-Net attaches a lightweight refinement stage to the input-resolution output of a point-cloud backbone. The released PTv3 implementation contains the three components described in the paper:

- **Confidence-Guided Query Selection (CQS):** selects the lowest-confidence `query_ratio` of points and uses every non-query point as a candidate anchor.
- **Asymmetric Decoupled Attention (ADA):** forms query/key embeddings from normalized coordinates and point attributes, adds relative XYZ geometry, and uses deep semantic features only as values.
- **Local Geometric Reconstruction (LGR):** reconstructs the selected query's coordinate--attribute vector during training and is skipped during inference.

## Paper-to-code map

| Paper component | Implementation |
| --- | --- |
| AFA-Net | `pointcept/models/afanet.py::AFANet` |
| CQS | `ConfidenceGuidedQuerySelection` |
| ADA | `AsymmetricDecoupledAttention` |
| LGR | `LocalGeometricReconstruction` |
| Overall objective | `pointcept/models/losses/afanet_loss.py::AFANetLoss` |

The indoor input vector is `[normalized_xyz, rgb]` (`input_dim=6`). SemanticKITTI uses `[normalized_xyz, intensity]` (`input_dim=4`).

## Configurations

- `configs/s3dis/afanet_ptv3.py`
- `configs/scannet/afanet_ptv3.py`
- `configs/semantic_kitti/afanet_ptv3.py`

Default paper settings are `query_ratio=0.25`, `neighborhood_size=16`, `lambda_main=1.0`, `lambda_aux=1.0`, and `lambda_rec=10.0`. Mix3D is disabled in the provided configurations.

## Environment

The project targets Python 3.10, PyTorch 2.5, and CUDA 12.4. Create the environment and compile the bundled PointOps extension with:

```bash
conda env create -f environment.yml
conda activate pointcept-torch2.5.0-cu12.4
```

Prepare S3DIS, ScanNet-V2, or SemanticKITTI under the data paths used by the selected config. Sparse masks are controlled by `labeled_ratio`, `use_precomputed_mask`, and `mask_root` in each dataset config.

## Training and evaluation

For example, train and evaluate the S3DIS/PTv3 setting with:

```bash
sh scripts/train.sh -d s3dis -c afanet_ptv3 -n afanet_s3dis -g 1
sh scripts/test.sh -d s3dis -c afanet_ptv3 -n afanet_s3dis -g 1
```

Replace `s3dis` with `scannet` or `semantic_kitti` for the other released configurations.

## Acknowledgements

This repository builds on Pointcept and Point Transformer V3. Please follow the licenses and citation requirements of the upstream projects.

## License

See [LICENSE](LICENSE).
