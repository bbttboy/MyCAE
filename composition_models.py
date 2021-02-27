"""Models for Text and Image Composition"""

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions
from bert_serving.client import BertClient
from torch.autograd import Variable

bc = BertClient()


class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super().__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()
        # self.name = 'model_name'

    def extract_img_feature(self, imgs):
        raise NotImplementedError

    def extract_text_feature(self, text_query, use_bert):
        raise NotImplementedError

    def compose_img_text(self, imgs, text_query):
        raise NotImplementedError

    # def compute_loss(self,
    #                  imgs_query, 
    #                  text_query, 
    #                  imgs_target, 
    #                  soft_triplet_loss=True):
    #     dct_with_representations = self.compose_img_text(imgs_query, text_query)
    #     composed_source_image = self.normalization_layer(dct_with_representations["repres"])
    #     target_img_features_non_norm = self.extract_img_feature(imgs_target)
    #     target_img_features = self.normalization_layer(target_img_features_non_norm)
    #     assert (composed_source_image.shape[0] == target_img_features.shape[0] and
    #             composed_source_image.shape[1] == target_img_features.shape[1])

    #     CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(-1.0), requires_grad=False)
    #     conjugate_representations = self.compose_img_text_features(target_img_features_non_norm, 
    #                                     dct_with_representations["text_features"],
    #                                     CONJUGATE)
    #     composed_target_image = self.normalization_layer(conjugate_representations["repres"])
    #     source_img_features = self.normalization_layer(dct_with_representations["img_features"]) #img1


class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
    """Base class for image and text encoder."""
    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__()
        # img model
        # pretrained表示是否是预训练模型
        img_model = torchvision.models.resnet18(pretrained=True)
        
        class GlobalAvgPool2d(torch.nn.Module):
            def forward(self, x):
                # (1,1)表示输出的维度, 即求最后两个维度的平均值
                # 例如输入x.shape=(512, 3, 64, 64)，输出的shape=(512, 3, 1, 1)
                # 即求(64, 64)的平均值
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d()
        # fc --> full connection 全连接层，即线性层
        # Linear(in, out)两个参数表示输入和输出的维度
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, image_embed_dim))
        self.img_model = img_model

        # text model
        # 一般没用自写的TextModel，一般还是用bert
        self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab=text_query,
            word_embed_dim=text_embed_dim,
            lstm_hidden_dim=text_embed_dim)

    def extract_img_feature(self, imgs):
        return self.img_model(imgs)

    def extract_text_feature(self, text_query, use_bert):
        if use_bert:
            text_features = bc.encode(text_query)
            return torch.from_numpy(text_features).cuda()
        return self.text_model(text_query)


class ComplexProjectionModule(torch.nn.Module):
    """
    introduce: 复映射模块
    """
    def __init__(self, image_embed_dim=512, text_embed_dim=768):
        super().__init__()
        self.bert_features = torch.nn.Sequential(
            # BatchNorm1d中参数num_features说明, 即num_features对应 C 或 L
            # C from an expected input of size (N, C, L) or L from input of size (N, L)
            # 例如num_features=C， 输入shape=(N, C, L)， 则分别对C个(N, L)切片进行归一化
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Linear(text_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            # 最后输出的shape是image_embed_dim
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim), 
            torch.nn.Dropout(p=0.5), 
            torch.nn.ReLU(), 
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

        def forward(self, x):
            # x --> (img_features, text_features, CONJUGATE)
            x1 = self.image_features([0])
            x2 = self.bert_features(x[1])
            # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
            # which results in the CONJUGATE of text features in the complex space
            CONJUGATE = x[2]
            num_samples = x[0].shape[0]
            CONJUGATE = CONJUGATE[:num_samples]
            delta = x2 # text as rotation
            re_delta = torch.cos(delta)
            im_delta = CONJUGATE * torch.sin(delta)

            re_score = x1 * re_delta
            im_score = x2 * im_delta

            concat_x = torch.cat([re_score, im_score], 1)
            # squeeze --> 压缩; 
            # unsqueeze(d) --> d是扩充哪一维
            # 例如: a.shape=(2,3)  a.unsqueeze(-1) --> a.shape=(2, 3, 1)
            x0copy = x[0].unsqueeze(1)
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            re_score = re_score.unsqueeze(1)
            im_score = im_score.unsqueeze(1)

            return concat_x, x1, x2, x0copy, re_score, im_score


class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        theta_linear = self.mapping(x[0])
        return theta_linear


class ConvMapping(torch.nn.Module):
    """
    This is convoultional mapping to image space. rho_conv(.)
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        # Conv1d中kernel_size相当于滑动窗口的大小
        # in_channels=5, out_channels=64, padding是在输入外加一圈0
        self.conv = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveAvgPool1d(16)

    def forward(self, x):
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1024))
        theta_conv = self.mapping(final_vec)
        return theta_conv
        