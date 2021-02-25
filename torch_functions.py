import numpy as np 
import torch
import torchvision

def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matri
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """

    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


class NormalizationLayer(torch.nn.Module):
    """Class for normalization layer."""
    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        # normalize_scale-->缩放参数，即正则化比例
        self.norm_s = float(normalize_scale)
        # 是否是把norm_s作为Parameter进行学习
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s, )))

        def forward(self, x):
            # torch.norm中，'p'表示范数形式，不设置p表示默认为2范数
            # dim表示在哪个维度上进行计算，例如shape=(2,4)在dim=1上进行计算，输出是一个(2)的形式
            # keepdim=True表示维持原有Shape的形态，同上，输出是一个(2,1)的形式，如果在dim=0上计算，输出是(1,4)形式
            # expand_as(x)表示对tensor进行广播扩充成同x的形式，即把(2,1)或(1,4)广播扩充成(2,4)
            features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
            return features