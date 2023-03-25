from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('partitions/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s = deepcopy(img), deepcopy(img)

        img_w = normalize(img_w)

        if random.random() < 0.5:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = transforms.RandomGrayscale(p=0.5)(img_s)
        img_s = blur(img_s, p=0.5)

        cutmix_box = obtain_cutmix_box(img_s.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s, ignore_mask = normalize(img_s, ignore_mask)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return img_w, img_s, ignore_mask, cutmix_box

    def __len__(self):
        return len(self.ids)
