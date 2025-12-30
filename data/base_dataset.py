import logging
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    """
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.logger = logging.getLogger("geoprop")
        
        self.num_points = cfg['dataset']['num_points']
        self.block_size = cfg['dataset']['block_size']
        
        self.aug_cfg = None
        if split == 'train' and 'augmentation' in cfg['train']:
            self.aug_cfg = cfg['train']['augmentation']

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")