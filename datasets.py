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
        # 拿到文件夹'D:\DataSet\fashion-200k\labels'内所有的文件路径
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        # 筛选包含 split='train' 字符串的文件路径
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []

        # 此处是否需要在转义单词前加一个空格？ 例如：将'dotmark'改为 ' dotmark'
        def caption_post_process(s):
            return s.strip().replace('.', 'dotmark').replace(
                '?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')
        
        for filename in label_files:
            print('read: ' + filename)
            with open(label_path + '/' + filename) as f:
                lines = f.readlines()
            for line in lines:
                # 此处应该是'\t'
                # 并且captions处貌似应该使用split(' ')   
                # 不用split()， 直接调用的get_different_word里面会有用split()
                line = line.split(' ')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    # captions是指图像的属性，即图像说明文字
                    # 这里有 hervé léger 这样的字体 要注意字符解码，会不会在上面readlines()就需要进行？
                    'captions': [caption_post_process(line[2])],  
                    'split': split,
                    'modifiable': False,
                }
                self.imgs += [img]
        print('Fashion200k: ', len(self.imgs), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()

    def get_different_word(self, source_caption, target_caption):
        """通过两张图像属性差异的部分，构建replace A to B的形式."""
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str 

    def generate_test_queries_(self):
        file2imgid = {}
        # 生成与图片路径对应的id，即第一个图片路径id=0，+1类推
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            # 从这一句看，test_quries.txt 里面存储的是源图片路径和目标图片路径
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[dix]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'mod': {
                    'str': mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        # 因为图片说明有许多重复的
        # 所以给不同的图片说明编号，并将相同图片说明对应的imgid放入对应的列表
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['caption']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
            self.caption2imgids = caption2imgids
            print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        # 产生父词，即只比子词少一个单词
        # parent2children_captions中的 key --> 父词； value -- > list(所有的子词)
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                # 考虑多个空格时 该句会有问题， 试一试 ' '.join(p.split()).strip()
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.imgs:
            img['modifiale'] = False
            # 父词列表 --> 键值对
            img['parent_captions'] = []
        # 为每个img字典对象中添加其可以对应的父词
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:  # 子词至少有2个，即可以进行修改
                for c in parent2children_captions[p]:  # c 是 父词p 对应的子词
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        
        # 有父词的img就是可以修改的img
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images ', num_modifiable_imgs)
    
    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target images (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['caption'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str