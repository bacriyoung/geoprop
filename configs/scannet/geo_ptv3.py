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
# [Reference] Pointcept ScanNet Official: 800 epochs
epoch_num = 100
epoch = epoch_num 
eval_epoch = epoch_num 

# [Reference] Pointcept ScanNet Official: Batch Size 12 (Total)
# Adjust 'batch_size' based on your GPU memory
batch_size = 2  
num_worker = 4 
save_freq = None

# [AMP] Enable Mixed Precision
enable_amp = True 
empty_cache = False

# [Critical] Disable Mix3D for GeoProp (Weak Supervision requirement)
mix_prob = 0.0 

# Dataset Parameters
num_classes = 20
ignore_index = 255 

# -------------------------------------------------------------------------
# Model Settings (GeoPTV3)
# -------------------------------------------------------------------------
model = dict(
    type="GeoPTV3",
    geo_input_dim=6,
    num_classes=num_classes,
    # [Config] Crop size for JAFAR module
    num_points=102400, 
    criteria=dict(
        type="GeoCoTrainLoss", 
        # [Correction] Restored to your S3DIS weak supervision weights
        lambda_main=10.0, 
        lambda_aux=4.0,   
        lambda_aff=1.0, 
        lambda_rec=20.0, # High reconstruction weight for weak supervision
        lambda_dist=0.1,
        lambda_bdy=0.1,
        warmup_epochs=10, 
        ignore_index=ignore_index,
        class_weights=[1.00, 1.00, 1.17, 1.75, 1.00, 1.94, 1.35, 1.04, 1.18, 1.96, 9.16, 9.28, 2.49, 2.18, 9.38, 10.00, 10.00, 10.00, 10.00, 1.38] 
    ),
    backbone_ptv3_cfg=dict(
        type="PointTransformerV3",
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
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
)

# -------------------------------------------------------------------------
# Optimizer & Scheduler
# -------------------------------------------------------------------------
# [Reference] Pointcept ScanNet Official: LR = 0.006
lr = 0.001 
clip_grad = 10.0
optimizer = dict(
    type="AdamW", 
    lr=lr, 
    weight_decay=0.05,
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr * 0.1], 
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

param_dicts = [dict(keyword="sem_stream", lr=lr * 0.1)]

# -------------------------------------------------------------------------
# Data Settings
# -------------------------------------------------------------------------
data = dict(
    num_workers=num_worker,
    batch_size=batch_size, 
    batch_size_val=1,           
    batch_size_test=1,          
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=[
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
        "window", "bookshelf", "picture", "counter", "desk", "curtain",
        "refridgerator", "shower curtain", "toilet", "sink", "bathtub",
        "otherfurniture",
    ],
    
    # -------------------------------------------------------------
    # Train: ScanNetGeoDataset with KNN Crop
    # -------------------------------------------------------------
    train=dict(
        type="ScanNetGeoDataset",
        split="train",
        data_root="data/scannet",
        num_points=102400, 
        voxel_size=0.02,
        transform=None,   
        loop=8, 
        labeled_ratio=0.001, 
        test_mode=False,
        ignore_index=ignore_index,
        rot_z_range=[-1, 1],
        tilt_range=[-1/64, 1/64],
        scale_range=[0.9, 1.1],
        jitter_sigma=0.005,
        color_drop_prob=0.2,
        chromatic_autocontrast_p=0.2,
        chromatic_translation_p=0.95,
        chromatic_translation_ratio=0.05,
        chromatic_jitter_std=0.05,
    ),

    # -------------------------------------------------------------
    # Val: Sliding Window
    # -------------------------------------------------------------
    val=dict(
        type="ScanNetGeoDataset",
        split="val",
        data_root="data/scannet",
        num_points=102400, 
        voxel_size=0.02,
        test_mode=True,   
        scan_mode='xy',
        stride=4.0,
        transform=None,
        loop=1,
        ignore_index=ignore_index,
        tta_conf=dict(enable=False)
    ),

    # -------------------------------------------------------------
    # Test: Sliding Window 
    # -------------------------------------------------------------
    test=dict(
        type="ScanNetGeoDataset",
        split="val", 
        data_root="data/scannet",
        num_points=102400,
        voxel_size=0.02,
        test_mode=True, 
        scan_mode='xyz',
        stride=1.5,       
        transform=None,
        ignore_index=ignore_index,
        tta_conf=dict(
            enable=True,
            scales=[0.95, 1.05],
            rotations=[0, 1, 2, 3],
            flip=True
        )
    ),
)

# -------------------------------------------------------------------------
# Test Config
# -------------------------------------------------------------------------
test_cfg = dict()

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
    dict(type="PreciseEvaluator", test_last=False),
]