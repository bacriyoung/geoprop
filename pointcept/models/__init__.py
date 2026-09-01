from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier
from .modules import PointModule, PointModel

# Backbones (只保留 PTv3)
# from .sparse_unet import *
# from .point_transformer import *
# from .point_transformer_v2 import *
from .point_transformer_v3 import *
# from .stratified_transformer import *
# from .spvcnn import *
# from .octformer import *
# from .oacnns import *

# AFA-Net semantic segmentation
from .afanet import AFANet

# Instance Segmentation (全部注释/删除)
# from .point_group import *
# from .sgiformer import *

# Pretraining (全部注释/删除)
# from .masked_scene_contrast import *
# from .point_prompt_training import *
# from .sonata import *
# from .concerto import *
