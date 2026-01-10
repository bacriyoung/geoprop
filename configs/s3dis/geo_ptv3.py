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
epoch_num = 100 
epoch = epoch_num 
eval_epoch = epoch_num 
# [Attention] 
batch_size = 2  
num_worker = 4 
save_freq = None

# [AMP] Enable Mixed Precision
enable_amp = True 
empty_cache = False
mix_prob = 0.0 # [Critical] Disable Mix3D for weak supervision

# Dataset Parameters
num_classes = 13 
ignore_index = 255 

# -------------------------------------------------------------------------
# Model Settings (GeoPTV3)
# -------------------------------------------------------------------------
model = dict(
    type="GeoPTV3",
    geo_input_dim=6,
    num_classes=num_classes,
    num_points=180000, 
    criteria=dict(
        type="GeoCoTrainLoss", 
        lambda_main=10.0, 
        lambda_aux=4.0,   
        lambda_aff=0.1, 
        lambda_dist=0.1,
        lambda_bdy=0.1,
        warmup_epochs=15, 
        ignore_index=ignore_index
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
lr = 0.001 
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
# Data Settings (Modified for Consistency)
# -------------------------------------------------------------------------
data = dict(
    num_workers=num_worker,
    batch_size=batch_size, 
    batch_size_val=1,           
    batch_size_test=1,          
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=[
        "ceiling", "floor", "wall", "beam", "column", "window", "door",
        "table", "chair", "sofa", "bookcase", "board", "clutter"
    ],
    
    # -------------------------------------------------------------
    # Train: Random KNN Crop 
    # -------------------------------------------------------------
    train=dict(
        type="S3DISCoTrainDataset",
        split="train",
        data_root="data/s3dis",
        num_points=180000, 
        voxel_size=0.02,
        transform=None,   
        loop=30,
        labeled_ratio=0.001,
        test_mode=False,
    ),

    # -------------------------------------------------------------
    # Val: Sliding Window 
    # -------------------------------------------------------------
    val=dict(
        type="S3DISCoTrainDataset",
        split="val",
        data_root="data/s3dis",
        num_points=180000, 
        voxel_size=0.02,
        test_mode=True,   
        stride=2.0,       
        transform=None,
        loop=1,  
    ),

    # -------------------------------------------------------------
    # Test: Sliding Window 
    # -------------------------------------------------------------
    test=dict(
        type="S3DISCoTrainDataset",
        split="Area_5",
        data_root="data/s3dis",
        num_points=180000,
        voxel_size=0.02,
        test_mode=True, 
        stride=2.0,       
        transform=None,   
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