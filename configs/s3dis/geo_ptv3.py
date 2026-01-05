_base_ = ["../_base_/default_runtime.py"]

# -------------------------------------------------------------------------
# Global Variables & Hyper-parameters
# -------------------------------------------------------------------------
weight = None
resume = False
evaluate = True
test_only = False
seed = 38345489

# 训练参数
epoch_num = 100 
batch_size = 4
num_worker = 4 # ⬆️ 提升到8，加快数据读取
save_freq = 5

# 🔴 [AMP] 开启混合精度，训练更快更省显存
enable_amp = True 
empty_cache = False
mix_prob = 0.0 # 🔴 [Critical] 弱监督必须关掉 Mix3D，否则几何流会崩

# 数据集参数
num_classes = 13 
# 注意：Evaluator 读取的是 data.ignore_index，Loss 读取的是 model.criteria.ignore_index
ignore_index = 255 

# -------------------------------------------------------------------------
# Model Settings (Aligned with Official PTv3m1)
# -------------------------------------------------------------------------
model = dict(
    type="GeoPTV3",
    geo_input_dim=6,
    num_classes=num_classes,
    num_points=80000,
    criteria=dict(
        type="GeoCoTrainLoss", 
        lambda_sup=10.0,  # 0.1% 场景下的强监督锚点
        lambda_con=1.0, 
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
        enc_channels=(32, 64, 128, 256, 512), # ✅ 对齐官方
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),      # ✅ 对齐官方
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
# Optimizer & Scheduler (Hybrid Strategy)
# -------------------------------------------------------------------------
# 🔴 学习率策略：Backbone 慢(0.1x)，Head 快(1x)
# 这能防止 PointJAFAR 的噪音初期把 Backbone 搞坏
lr = 0.001 # 略微提高一点点，因为有了 param_dicts 保护 backbone
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.05)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr * 0.1], # Head 用 0.002, Backbone 用 0.0002
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# 这里定义哪些参数属于 backbone (享受 0.1x 学习率)
param_dicts = [dict(keyword="sem_stream", lr=lr * 0.1)]

# -------------------------------------------------------------------------
# Data Settings
# -------------------------------------------------------------------------
data = dict(
    num_workers=num_worker,
    batch_size=batch_size,
    num_classes=num_classes,
    ignore_index=ignore_index, 
    names=[
        "ceiling", "floor", "wall", "beam", "column", "window", "door",
        "table", "chair", "sofa", "bookcase", "board", "clutter"
    ],
    
    train=dict(
        type="S3DISCoTrainDataset",
        split="train",
        data_root="data/s3dis",
        num_points=80000,
        voxel_size=0.02,
        transform=None, # 使用 dataset 内部的 crop
        loop=30,
        labeled_ratio=0.001, # 0.1% 标注
    ),
    val=dict(
        type="S3DISCoTrainDataset",
        split="val",
        data_root="data/s3dis",
        num_points=80000,
        voxel_size=0.02,
        test_mode=True,
        transform=None,
        loop=1, 
    ),
)

# -------------------------------------------------------------------------
# Hooks
# -------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=100),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=save_freq),
]