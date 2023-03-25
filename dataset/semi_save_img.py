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

        # img = img.resize((321, 321), Image.BILINEAR)
        # mask = mask.resize((321, 321), Image.NEAREST)
        # ignore_value = 254 if self.mode == 'train_u' else 255
        # img, mask = crop(img, mask, 321, ignore_value)
        # img, mask = hflip(img, mask, p=0.5)
        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask, id

    def __len__(self):
        return len(self.ids)
