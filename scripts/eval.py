"""
Visualize which images have low and high confidence scores in training set.
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dotmap import DotMap
from src.utils import utils
from torchvision import transforms
from torch.utils.data import DataLoader
from src.systems.simclr import SimCLRSystem, TaU_SimCLRSystem
from src.systems.mocov2 import MoCoV2System, TaU_MoCoV2System
from src.systems.transfer import TransferSystem, Pretrained_TaU_SimCLRSystem
from torchvision.utils import save_image, make_grid
from src.datasets.utils import DICT_ROOT
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.imagenet import ImageNet
from src.datasets.mscoco import MSCOCO
from src.datasets.transforms import get_transforms
from src.datasets.transforms import IMAGE_SHAPE, CROP_SHAPE


@torch.no_grad()
def viz(exp_dir, checkpoint_name, dataset, gpu_device=-1):
    config_path = os.path.join(exp_dir, 'config.json')
    config_json = utils.load_json(config_path)
    config = DotMap(config_json)
    config.dataset.name = dataset
    if gpu_device >= 0:
        config.gpu_device = gpu_device
        config.cuda = True
        device = torch.device(f'cuda:{gpu_device}')
    else:
        config.cuda = False
        device = torch.device('cpu')

    SystemClass = globals()[config.system]
    system = SystemClass(config)
    checkpoint_file = os.path.join(exp_dir, 'checkpoints', checkpoint_name)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    system.load_state_dict(checkpoint['state_dict'])
    system.config = config
    system = system.eval()
    system = system.to(device)

    norms = []
    labels = []
    with torch.no_grad():
        val_loader = DataLoader(system._val_dataset, batch_size=1, shuffle=False, num_workers=8)
        pbar = tqdm(total=len(val_loader))
        for batch in val_loader:
            image, label = batch[1], batch[-1]
            image = image.to(device)
            print("*****FORWARD")
            mean, score = system.forward(image)
            print("*****end")
            norms.append(score.squeeze(-1).cpu().numpy())
            labels.append(label[0].cpu().numpy())
            pbar.update()
        pbar.close()

    norms = np.concatenate(norms)
    labels = np.concatenate(labels).astype(int)
    dataset = get_datasets(config.dataset.name, train=False)

    return 0, 0


def get_datasets(name, train=True):
    image_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE[name]),
        transforms.CenterCrop(CROP_SHAPE[name]),
        transforms.ToTensor(),
    ]) 
    root = DICT_ROOT[name]

    if name == 'cifar10':
        dataset = CIFAR10(root, train=train, image_transforms=image_transforms)
    elif name == 'cifar100':
        dataset = CIFAR100(root, train=train, image_transforms=image_transforms)
    elif name == 'stl10':
        dataset = STL10(root, train=train, image_transforms=image_transforms)
    elif name == 'tinyimagenet' or name == 'imagenet':
        dataset = ImageNet(root, train=train, image_transforms=image_transforms)
    elif name == 'mscoco':
        dataset = MSCOCO(root, train=train, image_transforms=image_transforms)
    else:
        raise Exception(f'Dataset {name} not supported.')

    return dataset



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, default='experiment directory')
    parser.add_argument('checkpoint_name', type=str, help='checkpoint name')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()

    out_dir = os.path.join(args.exp_dir, 'viz')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    grid_low, grid_high = viz(
        args.exp_dir, 
        checkpoint_name=args.checkpoint_name,
        dataset=args.dataset,
        gpu_device=args.gpu_device)