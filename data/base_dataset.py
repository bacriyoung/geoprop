import logging
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    所有数据集的基类 (Abstract Base Class)。
    负责统一的初始化逻辑，如 Logger 获取、配置保存等。
    """
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        # 所有子类自动拥有 self.logger
        self.logger = logging.getLogger("geoprop")
        
        # 解析通用配置
        self.num_points = cfg['dataset']['num_points']
        self.block_size = cfg['dataset']['block_size']
        
        # 仅在训练且开启增强时获取增强配置
        self.aug_cfg = None
        if split == 'train' and 'augmentation' in cfg['train']:
            self.aug_cfg = cfg['train']['augmentation']

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")