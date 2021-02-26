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

        img_model = torchvision.models.resnet18(pretrained=True)
        
        class GlobalAvgPool2d(torch.nn.Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool