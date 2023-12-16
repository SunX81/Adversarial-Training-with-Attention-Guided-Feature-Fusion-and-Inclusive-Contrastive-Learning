import torch
from torchvision import datasets, transforms
import torch.nn as nn
import logging
import os
from collections import OrderedDict
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import numpy as np
import random
from collections import defaultdict

logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# The targeted adversarial patches generated on four datasets ImageNet, ImageNette, CIFAR, and CIFAR100
patch6_net = np.load('./patches/net-vit/patch6/best_patch.npy')
patch7_net = np.load('./patches/net-vit/patch7/best_patch.npy')
patch8_net = np.load('./patches/net-vit/patch8/best_patch.npy')

patch6_nette = np.load('./patches/nette-vit/patch6/best_patch.npy')
patch7_nette = np.load('./patches/nette-vit/patch7/best_patch.npy')
patch8_nette = np.load('./patches/nette-vit/patch8/best_patch.npy')

patch6_cifar = np.load('./patches/cifar-vit/patch6/best_patch.npy')
patch7_cifar = np.load('./patches/cifar-vit/patch7/best_patch.npy')
patch8_cifar = np.load('./patches/cifar-vit/patch8/best_patch.npy')

patch6_cifar100 = np.load('./patches/cifar100-vit/patch6/best_patch.npy')
patch7_cifar100 = np.load('./patches/cifar100-vit/patch7/best_patch.npy')
patch8_cifar100 = np.load('./patches/cifar100-vit/patch8/best_patch.npy')


def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 224, 224)):
    """Generate the mask and apply the patch"""

    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1] - patch.shape[1]), np.random.randint(
            low=0, high=image_size[2] - patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask


def patch_attack(patch, image):
    """Added the targeted patches to the images"""

    applied_patch, mask = mask_generation("rectangle", patch, image_size=(3, 224, 224))
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    adv = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
        (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    return adv


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, patch, transform):
        self.transform = transform
        self.patch = patch

    def __call__(self, x):
        x = self.transform(x)
        x_adv = patch_attack(self.patch, x)

        return [x, x_adv]


class PatchTransform:
    """Create two crops of the same image"""

    def __init__(self, patch, transform):
        self.transform = transform
        self.patch = patch

    def __call__(self, x):
        x = self.transform(x)
        x_adv = patch_attack(self.patch, x)
        return x_adv


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def get_loaders(args):
    if args.dataset == "cifar" or args.dataset == "cifar-un":
        mean = cifar10_mean
        std = cifar10_std
    if args.dataset == "cifar100" or args.dataset == "cifar100-un":
        mean = cifar100_mean
        std = cifar100_std
    elif args.dataset == "imagenette" or args.dataset == "imagenet" or args.dataset == "imagenet-un" or args.dataset == "imagenette-un":
        mean = imagenet_mean
        std = imagenet_std

    train_list = [
        transforms.Resize([args.resize, args.resize]),
        transforms.RandomCrop(args.crop, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    train_list.append(transforms.ToTensor())
    train_list.append(transforms.Normalize(mean, std))
    train_transform = transforms.Compose(train_list)
    test_transform = transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    notar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    num_workers = 10

    if args.dataset == "cifar":
        if args.kind == "patch8":
            patch = patch8_cifar
        elif args.kind == "patch7":
            patch = patch7_cifar
        else:
            patch = patch6_cifar

        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                          transform=TwoCropTransform(patch, train_transform))
        test_patch_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                               transform=PatchTransform(patch, test_transform))
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch_loader = torch.utils.data.DataLoader(
            dataset=test_patch_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, [], test_loader, test_patch_loader

    if args.dataset == "cifar-un":
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, [], test_loader, []

    if args.dataset == "cifar100":
        if args.kind == "patch8":
            patch = patch8_cifar100
        elif args.kind == "patch7":
            patch = patch7_cifar100
        else:
            patch = patch6_cifar100

        train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=TwoCropTransform(patch, train_transform))
        test_patch_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=PatchTransform(patch, test_transform))
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch_loader = torch.utils.data.DataLoader(
            dataset=test_patch_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, [], test_loader, test_patch_loader

    if args.dataset == "cifar100-un":
        train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, [], test_loader, []

    if args.dataset == "imagenette":
        if args.kind == "patch8":
            patch = patch8_nette
        elif args.kind == "patch7":
            patch = patch7_nette
        else:
            patch = patch6_nette
        train_dataset = datasets.ImageFolder(args.data_dir + "train/", TwoCropTransform(patch, train_transform))
        test_patch_dataset = datasets.ImageFolder(args.data_dir + "val/", PatchTransform(patch, test_transform))
        test_dataset = datasets.ImageFolder(args.data_dir + "val/", test_transform)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch_loader = torch.utils.data.DataLoader(
            dataset=test_patch_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, [], test_loader, test_patch_loader

    if args.dataset == "imagenette-un":
        train_dataset = datasets.ImageFolder(args.data_dir + "train/", train_transform)
        train_patch_dataset = datasets.ImageFolder(args.data_dir + "train-un/", notar_transform)
        test_dataset = datasets.ImageFolder(args.data_dir + "val/", test_transform)
        test_patch_dataset = datasets.ImageFolder(args.data_dir + "val-un/", notar_transform)

        torch.manual_seed(seed=0)
        g = torch.Generator()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            pin_memory=True,
            num_workers=num_workers,
        )

        torch.manual_seed(seed=0)
        g = torch.Generator()
        train_patch_loader = torch.utils.data.DataLoader(
            dataset=train_patch_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch_loader = torch.utils.data.DataLoader(
            dataset=test_patch_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, train_patch_loader, test_loader, test_patch_loader

    if args.dataset == "imagenet":
        if args.kind == "patch8":
            patch = patch8_net
        elif args.kind == "patch7":
            patch = patch7_net
        else:
            patch = patch6_net
        train_dataset = datasets.ImageFolder(args.data_dir + "train/", TwoCropTransform(patch, train_transform))
        test_patch_dataset = datasets.ImageFolder(args.data_dir + "val/", PatchTransform(patch, test_transform))
        test_dataset = datasets.ImageFolder(args.data_dir + "val/", test_transform)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch_loader = torch.utils.data.DataLoader(
            dataset=test_patch_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, [], test_loader, test_patch_loader

    if args.dataset == "imagenet-un":
        train_dataset = datasets.ImageFolder(args.data_dir + "train/", train_transform)
        train_patch_dataset = datasets.ImageFolder(args.data_dir + "train-un/", notar_transform)
        test_dataset = datasets.ImageFolder(args.data_dir + "val/", test_transform)
        test_patch_dataset = datasets.ImageFolder(args.data_dir + "val-un/", notar_transform)
        
        torch.manual_seed(seed=0)
        g = torch.Generator()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            pin_memory=True,
            num_workers=num_workers,
        )

        torch.manual_seed(seed=0)
        g = torch.Generator()
        train_patch_loader = torch.utils.data.DataLoader(
            dataset=train_patch_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch_loader = torch.utils.data.DataLoader(
            dataset=test_patch_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_loader, train_patch_loader, test_loader, test_patch_loader


_logger = logging.getLogger(__name__)


class MultiAverageMeter(object):
    """Computes and stores the average and current value for multiple metrics"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum_meter = defaultdict(float)
        self.lasts = defaultdict(float)
        self.counts_meter = defaultdict(int)
    def update(self, key, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.lasts[key] = val
        self.sum_meter[key] += val * n
        self.counts_meter[key] += n
    def last(self, key):
        return self.lasts[key]
    def avg(self, key):
        if self.counts_meter[key] == 0:
            return 0.0
        else:
            return self.sum_meter[key] / self.counts_meter[key]
    def __repr__(self):
        s = ""
        for k in self.sum_meter:
            s += "{}={:.4f} ".format(k, self.avg(k))
        return s.strip()
