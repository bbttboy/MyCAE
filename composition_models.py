"""Models for Text and Image Composition"""

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions
# from bert_serving.client import BertClient
from bert_dic import MyBertDic
from torch.autograd import Variable

# bc = BertClient()


class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super().__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()
        # self.name = 'model_name'

    def extract_img_feature(self, imgs):
        raise NotImplementedError

    def extract_text_feature(self, text_query, use_bert):
        raise NotImplementedError

    def compose_img_text(self, imgs, text_query):
        raise NotImplementedError

    def compute_loss(self,
                     imgs_query, 
                     text_query, 
                     imgs_target, 
                     soft_triplet_loss=True):
        # dct --> discrete consine transform(离散余弦变换)？
        """
        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.textdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features,
                                    }
        """
        dct_with_representations = self.compose_img_text(imgs_query, text_query)
        composed_source_image = self.normalization_layer(dct_with_representations["repres"])
        target_img_features_non_norm = self.extract_img_feature(imgs_target)
        target_img_features = self.normalization_layer(target_img_features_non_norm)
        assert (composed_source_image.shape[0] == target_img_features.shape[0] and
                composed_source_image.shape[1] == target_img_features.shape[1])
        # Get Rot_Sym_Loss --> Rotational Symmetry Loss(旋转对称损失)
        # 此处的name取自子类
        if self.name == 'composeAE':
            # conjugate --> 共轭
            CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(-1.0), requires_grad=False)
            conjugate_representations = self.compose_img_text_features(target_img_features_non_norm, 
                                            dct_with_representations["text_features"],
                                            CONJUGATE)
            composed_target_image = self.normalization_layer(conjugate_representations["repres"])
            source_img_features = self.normalization_layer(dct_with_representations["img_features"]) #img1
            if soft_triplet_loss:
                # rot_sym_loss --> Rotational Symmetry Loss --> 旋转对称损失
                dct_with_representations["rot_sym_loss"] = \
                    self.compute_soft_triplet_loss_(composed_target_image, source_img_features)
            else:
                dct_with_representations["rot_sym_loss"] = \
                    self.compute_batch_based_classification_loss_(composed_target_image, 
                                                                  source_img_features)
        else:  # tirg, RealSpaceConcatAE etc
            dct_with_representations["rot_sym_loss"] = 0

        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(composed_source_image, target_img_features), \
                dct_with_representations
        else:
            return self.compute_batch_based_classification_loss_(composed_source_image, target_img_features), \
                dct_with_representations
    
    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)
 
class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
    """Base class for image and text encoder."""
    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__()
        # img model
        # pretrained表示是否是预训练模型
        img_model = torchvision.models.resnet18(pretrained=True)
        self.name = name
        
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
        if use_bert:
            self.text_model = MyBertDic()
        else:
            self.text_model = text_model.TextLSTMModel(
                texts_to_build_vocab=text_query,
                word_embed_dim=text_embed_dim,
                lstm_hidden_dim=text_embed_dim)
        
    def extract_img_feature(self, imgs):
        """
        对应 z = Psi(x), x -- > image
        """
        return self.img_model(imgs)

    def extract_text_feature(self, text_query, use_bert):
        """
        对应 q = Beta(t), t --> text
        """
        if use_bert:
            text_features = self.text_model.encode(text_query)
            return torch.from_numpy(text_features).cuda()
            # text_features = bc.encode(text_query)
            # return torch.from_numpy(text_features).cuda()
        else:
            return self.text_model(text_query)


class ComplexProjectionModule(torch.nn.Module):
    """
    introduce: 复映射模块\n
    即对应 Delta = e{j * Gamma(q)}\n
    Phi = Deta * Eta(z)
    """
    def __init__(self, image_embed_dim=512, text_embed_dim=768):
        super().__init__()
        # 此处的bert_features函数对应 Gamma(q)
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

        # 此处的image_features函数对应 Eta(z)
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim), 
            torch.nn.Dropout(p=0.5), 
            torch.nn.ReLU(), 
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        # x --> (img_features, text_features, CONJUGATE)
        x1 = self.image_features(x[0])  # x[0]即 z, x1 = Eta(z)
        x2 = self.bert_features(x[1])  # x[1]即 q, x2 = Gamma(q)
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]  # CONJUGATE对应 j
        delta = x2 # text as rotation

        # (re_deta + im_delta) 对应 Delta
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        # (re_score + im_score) 对应 Phi
        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)  # 对应 Phi
        # squeeze --> 压缩; 
        # unsqueeze(d) --> d是扩充哪一维
        # 例如: a.shape=(2,3)  a.unsqueeze(-1) --> a.shape=(2, 3, 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        # (Phi, Eta(z), Gamma(q), z, re_Phi, im_Phi)
        return concat_x, x1, x2, x0copy, re_score, im_score


class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)\n
    即 Rho(Phi)
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        # mapping 即 Rho(Phi)
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
    This is convoultional mapping to image space. rho_conv(.)\n
    即对应 Rho_conv(Phi, z, q) 
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        # mapping + conv + adaptivepooling  即对应 Rho_conv 
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


class ComposeAE(ImgEncoderTextEncoderBase):
    """The ComposeAE model.

    The method is described in
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(
            ComplexProjectionModule(),
            LinearMapping()
        )
        
        self.encoderWithConv = torch.nn.Sequential(
            ComplexProjectionModule(),
            ConvMapping()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

        self.textdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim),
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)  # z
        text_features = self.extract_text_feature(text_query, self.use_bert)  # q

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features, 
                                    CONJUGATE=Variable(torch.cuda.FloatTensor(32, 1).fill_(1.0), requires_grad=False),):
        """
        即对应 Theta = f(z, q) = a * Rho(Phi) + b * Rho_conv(Phi, z, q)
        """
        theta_linear = self.encoderLinear((img_features, text_features, CONJUGATE))
        theta_conv = self.encoderWithConv((img_features, text_features, CONJUGATE))
        theta = self.a[1] * theta_linear + self.a[0] * theta_conv

        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.textdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features,
                                    }

        return dct_with_representations