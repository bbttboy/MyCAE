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
import torchvision
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
    parser.add_argument('--comment', type=str,default='fashion200k_composeAE')
    parser.add_argument('--dataset', type=str, default='fashion200k')
    parser.add_argument('--dataset_path', type=str, default=r'D:/DataSet/fashion-200k/')
    parser.add_argument('--model', type=str, default='composeAE')
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--use_bert', type=bool, default=True)
    parser.add_argument('--use_complete_text_query', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--category_to_train', type=str, default='all')
    parser.add_argument('--num_iters', type=int, default=160000)
    parser.add_argument('--loss', type=str, default='soft_triplet')
    parser.add_argument('--loader_num_workers', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs/fashion200k/')  # ../是上级目录
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
    
    # 经过实验  model.img_model.parameters()中包括model.img_model.fc.parameters()
    # 同理  model.parameters() 应该也包括上面两个参数
    # 为什么要去重复的参数
    # 为什么params要分别取三个模型的parameters()
    for _, p1 in enumerate(params):  # remove duplicated params
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)
    # 运行完后 model中不包含img_mdoel和fc的参数，img_model中不包含fc的参数
    # why

    optimizer = torch.optim.SGD(params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
    return model, optimizer

def train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer):
    """Function for train loop"""
    print('Begin training.')
    print(len(trainset.test_queries), len(testset.test_queries))
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    # 适用：适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的
    torch.backends.cudnn.benchmark = True
    losses_tracking = {}
    it = 0  # 迭代次数
    epoch = -1
    tic = time.time()
    l2_loss = torch.nn.MSELoss().cuda()

    while it < opt.num_iters:
        epoch += 1

        # show/log stats
        # round(x, n) --> n表示保留小数点后n位(四舍五入)
        # print(x, y, z)的写法可以不管x,y,z是不是str，同时会自动在x,y,z之间加空格
        print('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic, 4), opt.comment)

        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print('    Loss', loss_name, round(avg_loss, 4))
            logger.add_scalar(loss_name, avg_loss, it)
        # 通用api
        # 通用格式 add_something(tag name, object, iteration number)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)
        
        # 回收被销毁了但是没有被释放的循环引用的对象
        if epoch % 1 == 0:
            gc.collect()

        # test
        if epoch % 3 == 1:
            tests = []
            for name, dataset in [('train', trainset), ('test', testset)]:
                t = test_retrieval.test(opt, model, dataset)
                tests += [(name + ' ' + metric_name, metric_value) 
                        for metric_name, metric_value in t]
            for metric_name, metric_value in tests:
                logger.add_scalar(metric_name, metric_value, it)
                print('    ', metric_name, round(metric_value, 4))
        
        # save checkpoint
        torch.save({
                    'it': it, 
                    'opt': opt,
                    'model_state_dict': model.state_dict(),
                   }, 
                   logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

        # run training for 1 epoch
        model.train()
        trainloader = trainset.get_loader(
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)
        
        def training_1_iter(data):
            assert type(data) is list
            img1 = np.stack([d['source_img_data'] for d in data])
            img1 = torch.from_numpy(img1).float()
            img1 = torch.autograd.Variable(img1).cuda()

            img2 = np.stack([d['target_img_data'] for d in data])
            img2 = torch.from_numpy(img2).float()
            img2 = torch.autograd.Variable(img2).cuda()

            if opt.use_complete_text_query:
                if opt.dataset == 'mitstates':
                    supp_text = [str(d['noun']) for d in data]
                    mods = [str(d['mod']['str']) for d in data]
                    # text_query here means complete_text_query
                    text_query = [adj + " " + noun for adj, noun in zip(mods, supp_text)]
                else:
                    text_query = [str(d['target_caption']) for d in data]
            else:
                text_query = [str(d['mod']['str']) for d in data]
            # compute loss
            if opt.loss not in ['soft_triplet', 'batch_based_classification']:
                print('Invalid loss function', opt.loss)
                sys.exit()
            
            losses = []
            if_soft_triplet = True if opt.loss == 'soft_triplet' else False
            loss_value, dct_with_representations = model.compute_loss(img1,
                                                                      text_query,
                                                                      img2,
                                                                      soft_triplet_loss=if_soft_triplet,)

            loss_name = opt.loss
            losses += [(loss_name, loss_weights[0], loss_value.cuda())]

            if opt.model == 'composeAE':
                dec_img_loss = l2_loss(dct_with_representations["repr_to_compare_with_source"],
                                       dct_with_representations["img_features"])
                dec_text_loss = l2_loss(dct_with_representations["repr_to_compare_with_mods"],
                                        dct_with_representations["text_features"])
                
                losses += [("L2_loss", loss_weights[1], dec_img_loss.cuda())]  # loss_weight=0.1
                losses += [("L2_loss_text", loss_weights[2], dec_text_loss.cuda())]  # loss_weight=0.1
                # loss_weight=0.01
                losses += [("rot_sym_loss", loss_weights[3], dct_with_representations["rot_sym_loss"].cuda())]
            else:
                print("Invalid model.", opt.model)
                sys.exit()

            total_loss = sum([loss_weight * loss_value for loss_name, loss_weight, loss_value in losses])
            assert not torch.isnan(total_loss)
            losses += [('total training loss', None, total_loss.item())]

            # track losses
            for loss_name, loss_weight, loss_value in losses:
                if loss_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))
            
            torch.autograd.set_detect_anomaly(True)

            # gradient descendt
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # 在处理大规模数据时或者需要迭代多次耗时很长的任务时
        # 可以利用Python tqdm模块来显示任务进度条
        # tqdm使用方法：tqdm.tqdm(可迭代对象) ，括号中的可迭代对象可以是个list,tuple,dict等。
        # 这里直接使用tqdm没有用tqdm.tqdm是因为 from tqdm import tqdm
        # 即tqdm只是一个tqdm.py 需要从中 import tqdm函数
        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            it += 1
            training_1_iter(data)

            # decay learning rate
            if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1

    print('Finished training.')


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
    train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer)
    logger.close()

if __name__=="__main__":
    main()