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
    if args.dataset == "cifar":
        mean = cifar10_mean
        std = cifar10_std
    elif args.dataset == "imagenette" or args.dataset == "imagenet":
        mean = imagenet_mean
        std = imagenet_std
    else:
        mean = cifar100_mean
        std = cifar100_std

    test_transform = transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    notar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    num_workers = 12

    if args.dataset == "cifar":
        patch6 = patch6_cifar
        patch7 = patch7_cifar
        patch8 = patch8_cifar

        test_patch6_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                                transform=PatchTransform(patch6, test_transform))
        test_patch7_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                                transform=PatchTransform(patch7, test_transform))
        test_patch8_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                                transform=PatchTransform(patch8, test_transform))

        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch6_loader = torch.utils.data.DataLoader(
            dataset=test_patch6_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch7_loader = torch.utils.data.DataLoader(
            dataset=test_patch7_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch8_loader = torch.utils.data.DataLoader(
            dataset=test_patch8_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_untarget_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return test_loader, test_patch6_loader, test_patch7_loader, test_patch8_loader, test_untarget_loader

    elif args.dataset == "cifar100":

        patch6 = patch6_cifar100
        patch7 = patch7_cifar100
        patch8 = patch8_cifar100

        test_patch6_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                               transform=PatchTransform(patch6, test_transform))
        test_patch7_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                               transform=PatchTransform(patch7, test_transform))
        test_patch8_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                               transform=PatchTransform(patch8, test_transform))

        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        test_patch6_loader = torch.utils.data.DataLoader(
            dataset=test_patch6_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        test_patch7_loader = torch.utils.data.DataLoader(
            dataset=test_patch7_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        test_patch8_loader = torch.utils.data.DataLoader(
            dataset=test_patch8_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        test_untarget_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        return test_loader, test_patch6_loader, test_patch7_loader, test_patch8_loader, test_untarget_loader

    elif args.dataset == "imagenette" or args.dataset == "imagenet":
        if args.dataset == "imagenette":
            patch6 = patch6_nette
            patch7 = patch7_nette
            patch8 = patch8_nette
        else:
            patch6 = patch6_net
            patch7 = patch7_net
            patch8 = patch8_net

        test_patch6_dataset = datasets.ImageFolder(
            args.data_dir + "val/", PatchTransform(patch6, test_transform))
        test_patch7_dataset = datasets.ImageFolder(
            args.data_dir + "val/", PatchTransform(patch7, test_transform))
        test_patch8_dataset = datasets.ImageFolder(
            args.data_dir + "val/", PatchTransform(patch8, test_transform))

        test_dataset = datasets.ImageFolder(args.data_dir + "val/", test_transform)
        test_untarget_dataset = datasets.ImageFolder(args.data_dir + "val-un/", notar_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch6_loader = torch.utils.data.DataLoader(
            dataset=test_patch6_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch7_loader = torch.utils.data.DataLoader(
            dataset=test_patch7_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_patch8_loader = torch.utils.data.DataLoader(
            dataset=test_patch8_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_untarget_loader = torch.utils.data.DataLoader(
            dataset=test_untarget_dataset,
            batch_size=args.batch_size_eval // args.accum_steps * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return test_loader, test_patch6_loader, test_patch7_loader, test_patch8_loader, test_untarget_loader

    else:
        raise ValueError("Dataset doesn't exist！")


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
