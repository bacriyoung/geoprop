_base_ = ["../_base_/default_runtime.py"]

# -------------------------------------------------------------------------
# Global Variables & Hyper-parameters
# -------------------------------------------------------------------------
weight = None
resume = False
evaluate = True
test_only = False
seed = 38345489

# Training Parameters
epoch_num = 50 
epoch = epoch_num 
eval_epoch = epoch_num 

# Batch Size: Outdoor point clouds are large (120k points). 
# BS=2 or 4 is recommended for 24G/40G VRAM. 
batch_size = 2  
num_worker = 1 
save_freq = None

# [AMP] Enable Mixed Precision
enable_amp = True 
empty_cache = False

# [Critical] Disable Mix3D for GeoProp (Geometric Consistency)
mix_prob = 0.0 

# Dataset Parameters
num_classes = 19
ignore_index = 255

# Class Weights from PTv2 (Crucial for SemanticKITTI imbalance)
class_weights = [
    3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704, 10.1922, 
    1.6155, 4.2187, 1.9385, 5.5455, 2.0198, 2.6261, 1.3212, 5.1102, 
    2.5492, 5.8585, 7.3929
]

# -------------------------------------------------------------------------
# Model Settings (GeoPTV3)
# -------------------------------------------------------------------------
model = dict(
    type="GeoPTV3",
    # [Important] Input Dim = 6 (3 Coord + 3 Intensity replicated)
    geo_input_dim=6, 
    num_classes=num_classes,
    # Match PTv2 crop size
    num_points=120000, 
    criteria=dict(
        type="GeoCoTrainLoss", 
        lambda_main=10.0, 
        lambda_aux=1.0,   
        lambda_aff=1.0, 
        lambda_rec=20.0, # High weight for geometric reconstruction
        lambda_dist=0.1,
        lambda_bdy=0.1,
        warmup_epochs=5, 
        ignore_index=ignore_index, 
        class_weights=class_weights 
    ),
    backbone_ptv3_cfg=dict(
        type="PointTransformerV3",
        in_channels=6, # Coord + Intensity (3ch)
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
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D", "SemanticKITTI"),
    ),
)

# -------------------------------------------------------------------------
# Optimizer & Scheduler
# -------------------------------------------------------------------------
# LR optimized for Batch Size 2-4 (PTv2 used 0.002 for BS 8)
lr = 0.001 
optimizer = dict(
    type="AdamW", 
    lr=lr, 
    weight_decay=0.005, # PTv2 uses 0.005 for SemanticKITTI
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr * 0.1], 
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

param_dicts = [dict(keyword="sem_stream", lr=lr * 0.1)]

# -------------------------------------------------------------------------
# Data Settings
# -------------------------------------------------------------------------
dataset_type = "SemanticKITTIGeoDataset"
data_root = "data/semantic_kitti"

data = dict(
    num_workers=num_worker,
    batch_size=batch_size, 
    batch_size_val=1,           
    batch_size_test=1,          
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=[
        "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
        "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
        "other-ground", "building", "fence", "vegetation", "trunk",
        "terrain", "pole", "traffic-sign",
    ],
    
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        num_points=120000, 
        voxel_size=0.05,
        transform=None,   
        loop=1,
        labeled_ratio=0.001,
        ignore_index=ignore_index, 
        test_mode=False,
        rot_z_range=[-1, 1],
        tilt_range=[0, 0],
        scale_range=[0.9, 1.1],
        jitter_sigma=0.005,
        color_drop_prob=0.2,
        clip_range=[-35.2, -35.2, -4, 35.2, 35.2, 2],
        # Disable all chromatic augmentations for intensity
        chromatic_autocontrast_p=0.0,
        chromatic_translation_p=0.0,
        chromatic_translation_ratio=0.0,
        chromatic_jitter_std=0.0,
        # Use class-balanced sparse masks
        use_precomputed_mask=True,
        mask_root='data/semantic_kitti/masks/balanced_0.001',
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        num_points=120000, 
        voxel_size=0.05,
        test_mode=True,   
        scan_mode='xy',
        stride=4.0, # Larger stride for outdoor
        transform=None,
        loop=1,
        ignore_index=ignore_index,
        clip_range=[-35.2, -35.2, -4, 35.2, 35.2, 2],
        tta_conf=dict(enable=False)
    ),

    test=dict(
        type=dataset_type,
        split="test", 
        data_root=data_root,
        num_points=120000,
        voxel_size=0.05,
        test_mode=True, 
        scan_mode='xy',
        stride=2.0,       
        transform=None,
        ignore_index=ignore_index,
        clip_range=[-35.2, -35.2, -4, 35.2, 35.2, 2],
        tta_conf=dict(
            enable=True,
            scales=[0.95, 1.05],
            rotations=[0, 1, 2, 3],
            flip=True
        )
    ),
)

# -------------------------------------------------------------------------
# Hooks
# -------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=100),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=save_freq),
]