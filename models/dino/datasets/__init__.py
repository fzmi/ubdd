import torch.utils.data
import torchvision

from .xbd import build as build_xbd
    
def build_dataset(image_set, args):
    if args.dataset_file == 'xbd':
        return build_xbd(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
