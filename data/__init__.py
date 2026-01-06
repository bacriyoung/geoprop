from .base_dataset import BaseDataset
from .s3dis.s3dis_dataset import S3DISDataset
from .sensaturban.sensaturban_dataset import SensatUrbanDataset # [Add]
from .seeds import get_fixed_seeds

def build_dataset(cfg, split='train'):
    """
    Factory function to initialize the correct dataset based on config.
    """
    dataset_name = cfg['dataset']['name'].lower()
    
    if dataset_name == 's3dis':
        return S3DISDataset(cfg, split=split)
    elif dataset_name == 'sensaturban':
        return SensatUrbanDataset(cfg, split=split)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported in build_dataset factory.")