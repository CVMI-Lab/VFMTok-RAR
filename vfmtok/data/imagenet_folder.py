#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""
import numpy as np
import re, pdb, cv2
from PIL import Image
import os.path as osp
import torch, io, pdb
from glob import glob
from megfile import smart_open
from .register import set_aws_a
from torchvision import transforms
from einops import repeat, rearrange
from torch.utils.data import Dataset
from .augmentation import random_crop_arr, center_crop_arr

class ImageFolder(Dataset):
    """ImageNet dataset."""
    def __init__(self, img_folder = None, samples=None, transform=None):

        self.img_folder = img_folder
        self.transform = transform

        self.samples = None
        if (img_folder is not None):
            assert osp.exists(img_folder)
            self._retrive_imgs()
        else:
            assert samples is not None
            self.samples = samples
    
    def _retrive_imgs(self):

        self.samples = []
        suffix = ['.png', '.jpg', '.jpeg', ]
        
        for prefix in suffix:
            imgs = glob(osp.join(self.img_folder, '*' + prefix))
            self.samples.extend(imgs)



    def __getitem__(self, index):

        # Load the image
        img_file = self.samples[index]
        img = Image.open(img_file).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, img_file

    def __len__(self):

        return len(self.samples)

class ImageFolderDataset(ImageFolder):

    def __init__(self, anno_file, image_size, is_train = False, transform=None):

        super().__init__(anno_file,)
        if is_train:
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
            else:
                self.transform = transform

    def __getitem__(self, idx):

        image, target, nori_id = super().__getitem__(idx)
        return image, target, nori_id
