"""Provides data for training and testing."""

import numpy as np
import PIL
import skimage
import torch
import json
import torch.utils.data
import torchvision
import random
import warnings

class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self):
        super(BaseDataset, self).__init__
        self.imgs = []
        self.test_queries = []

    def get_loader(self, 
                   batch_size, 
                   shuffle=False, 
                   drop_last=False,
                   num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i,)

    def get_test_queries(self):
        return self.test_queries

    def get_all_texts(self):
        raise NotImplementedError("BaseDataset --> get_all_texts().")

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError("BaseDataset --> generate_random_query_target().")

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError("BaseDataset --> get_img().")


class Fashion200k(BaseDataset):
    """Fashion200k dataset."""

    def __init__(self, path, split='train', transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        # get label files for the split
        label_path = path + '/labels'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []

        def caption_post_process(s):
            return s.strip().replace('.', 'dotmark').replace(
                '?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')