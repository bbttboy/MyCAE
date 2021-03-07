import numpy as np
import torch
import random
from tqdm import tqdm 
from collections import OrderedDict

# def fiq_test(opt, model, testset):
#     # eval()一般用作测试集和验证集，即他再不会改变model的权重
#     model.eval()

#     all_imgs = []
#     all_queries = []
#     all_target_captions = []
#     all_target_ids = []

#     imgs = []
#     mods = []
#     out = []

#     for i in tqdm(range(len(testset))):
#         torch.cuda.empty_cache()
#         item = testset[i]
#         imgs += [testset.get_img(item['source_img_id'])]


def test(opt, model, testset):
    pass