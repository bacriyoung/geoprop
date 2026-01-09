_base_ = ["../_base_/default_runtime.py"]

# =========================================================================
# Global Variables & Hyper-parameters
# =========================================================================
weight = None
resume = False
evaluate = True
test_only = False
seed = 38345489

# Training Settings
epoch_num = 100 
epoch = epoch_num 
eval_epoch = epoch_num 
batch_size = 3  # Aligned with Official
num_worker = 4  # Aligned with Official
save_freq = None

# [Critical] Optimization for Structure-Aware Training
enable_amp = True 
empty_cache = False
# [MUST DISABLE] Mix3D ruins local geometric structure needed for JAFAR's boundary loss
mix_prob = 0.0 

# Dataset Constants
num_classes = 13 
ignore_index = 255 

# =========================================================================
# Model Settings (GeoPTV3 = PTv3 Backbone + JAFAR Stream)
# =========================================================================
model = dict(
    type="GeoPTV3", 
    
    # 1. JAFAR Specific Parameters
    # geo_input_channels=6,        # RGB + Voxel Center
    # jafar_kernel_size=5,         # Aligned with PTv3 Stem
    attn_search_radius=1,        # 3x3x3 Neighborhood search
    attn_dim=64,
    
    # 2. Loss Configuration (Internal)
    criteria=dict(
        type="GeoCoTrainLoss", 
        lambda_main=10.0,  # Semantic Loss (Strong supervision on 0.1% points)
        lambda_aux=4.0,   # PTv3 Aux Loss
        lambda_aff=0.1,   # Affinity Loss (Self-supervised)
        lambda_dist=0.1,  # Distribution Loss (Prototype alignment)
        lambda_bdy=0.1,   # Boundary Loss (Geometric Edge)
        warmup_epochs=15, 
        ignore_index=ignore_index
    ),

    # 3. PTv3 Backbone Parameters (Flattened here because GeoPTV3 inherits PTv3)
    in_channels=6,
    num_classes=num_classes,
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    # cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"), # Crucial for pre-trained weights
)

# =========================================================================
# Optimizer & Scheduler (Hybrid Strategy)
# =========================================================================
# Strategy: Backbone uses lower LR (0.1x) to preserve features; 
# JAFAR head uses higher LR (1x) to learn quickly from weak labels.
lr = 0.001 
optimizer = dict(
    type="AdamW", 
    lr=lr, 
    weight_decay=0.05,
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr * 0.1], # [Head, Backbone]
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Keyword match to identify backbone parameters
param_dicts = [dict(keyword="geo_stream", lr=lr)]

# =========================================================================
# Data Settings
# =========================================================================
data_root = "data/s3dis"
dataset_type = "S3DISDataset"

data = dict(
    num_workers=num_worker,
    batch_size=batch_size,
    num_classes=num_classes,
    ignore_index=ignore_index, 
    names=[
        "ceiling", "floor", "wall", "beam", "column", "window", "door",
        "table", "chair", "sofa", "bookcase", "board", "clutter"
    ],
    
    # Train: Custom Co-Train Dataset (Weak Supervision)
    train=dict(
        type="S3DISCoTrainDataset",
        split="train",
        data_root=data_root,
        # num_points=80000,
        voxel_size=0.02,
        loop=30,
        labeled_ratio=0.001, # 0.1% Labels
        transform=[
            dict(type="CenterShift", apply_z=True),
            # Data Augmentation
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # Voxelization (Generates 'grid_coord' needed by JAFAR)
            dict(
                type="GridSample",
                grid_size=0.02, # MUST match voxel_size
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # Cropping
            dict(type="SphereCrop", sample_rate=0.4, mode="random"),
            dict(type="SphereCrop", point_max=80000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
    ),

    # Val: Official Pipeline (Ensure Fair Comparison)
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train", # Official config uses mode='train' for val to keep consistency
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),

    # Test: Official Pipeline + TTA
    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [dict(type="RandomScale", scale=[0.9, 0.9]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[0.95, 0.95]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.05, 1.05]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.1, 1.1]), dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

# =========================================================================
# Hooks
# =========================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=100),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=save_freq),
    dict(type="PreciseEvaluator", test_last=False),
]