"""Main method to train the model."""

import argparse
import sys
import gc
import time
import datasets
import composition_models
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import test_retrieval
import torch
from tqdm import tqdm
from copy import deepcopy
import socket
import os
from datetime import datetime

torch.set_num_threads(3)

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='')
    parser.add_argument('--comment', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model', type=str, default='composeAE')
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--use_complete_text_query', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=9999999)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_dacay', type=float, default=1e-6)
    parser.add_argument('--category_to_train', type=str, default='all')
    parser.add_argument('--num_iters', type=int, default=160000)
    parser.add_argument('--loss', type=str, default='soft_triplet')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='../logs/')  # ../是上级目录
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--model_checkpoint', type=str, default='')

    args = parser.parse_args()
    return args

def load_dataset(opt):
    """Loads the input datasets."""
    print('Reading dataset: ', opt.dataset)
    if opt.dataset == 'fashion200k':
        trainset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                    # Resize 的输入为一个数的时候，会将维度最低的转换为输入的数，同时图像等比例缩放
                    # 也可以输入tuple，即转换为tuple大小
                    torchvision.transforms.Resize(224),
                    # CenterCrop(x) 即从中心位置裁剪一个输入大小 x 的图像
                    torchvision.transforms.CenterCrop(224),
                    # ToTensor会将PIL类型的图片(H, W, C) 转换为 (C, H, W)，同时除以255，即所有元素都在[0,1]
                    # 如果要将(C, H, W) 转换回 (H, W, C) 可以使用tensor的permute(1, 2, 0)
                    torchvision.transforms.ToTensor(),
                    # 此处输入的参数，第一个列表分别是3个通道的mean(均值)
                    # 第二个列表分别是3个通道的std(方差)
                    # Normalize --> output[channel] = (input[channel] - mean[channel]) / std[channel]
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        )
        testset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    else:
        print('Invalid dataset: ', opt.dataset)
        sys.exit()

    print('trainset size: ', str(len(trainset)))
    print('testset size: ', str(len(testset)))
    return trainset, testset

def create_model_and_optimizer(opt, texts):
    """Builds the model and related optimizer."""
    print("Creating model and optimizer for ", opt.model)
    text_embed_dim = 512 if not opt.use_bert else 768

    if opt.model == 'composeAE':
        model = composition_models.ComposeAE(texts,
                                             image_embed_dim=opt.image_embed_dim,
                                             text_embed_dim=text_embed_dim,
                                             use_bert=opt.use_bert,
                                             name=opt.model,)
    else:
        print('Invalid model: ', opt.model)
        sys.exit()

    model = model.cuda()

    # create optimizer
    params = [{
            'params': [p for p in model.img_model.fc.parameters()],
            'lr': opt.learning_rate,
        },{
            'params': [p for p in model.img_model.parameters()],
            'lr': 0.1 * opt.learning_rate,
        },{
            'params': [p for p in model.parameters()],
        }]
    
    for _, p1 in enumerate(params):  # remove duplicated params
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.SGD(params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
    return model, optimizer

def train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer):
    """Function for train loop"""
    

def main():
    opt = parse_opt()
    print('Arguments:')
    for k in opt.__dict__.keys():
        print('    ', k, ':', str(opt.__dict__[k]))

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')  # 月(英文)日_时-分-秒
    loss_weights = [1.0, 0.1, 0.1, 0.01]
    logdir = os.path.join(opt.log_dir, current_time + '_' + socket.gethostname() + opt.comment)

    logger = SummaryWriter(logdir)
    print('Log files saved to ', logger.file_writer.get_logdir())
    for k in opt.__dict__.keys():
        logger.add_text(k, str(opt.__dict__[k]))
    
    trainset, testset = load_dataset(opt)
    model, optimizer = create_model_and_optimizer(opt, [t for t in trainset.get_all_texts()])
    if opt.test_only:
        print('Doing test only!')
        checkpoint = torch.load(opt.model_chekpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        it = checkpoint['it']
        model.eval()
        tests = []
        it = 0
        for name, dataset in [('train', trainset), ('test', testset)]:
            t = test_retrieval.test(opt, model, dataset)
            tests += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t]
        for metric_name, metric_value in tests:
            logger.add_scalar(metric_name, metric_value, it)
            print('    ', metric_name, round(metric_value, 4))
        
        return 0
    train_loop(opt, loss_weights, logger, trainset， testset, model, optimizer)
    logger.close()