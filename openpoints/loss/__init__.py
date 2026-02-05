from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .distill_loss import  DistillLoss
from .build import build_criterion_from_cfg
from .geo_loss import GeoCoTrainLoss