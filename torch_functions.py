import numpy as np 
import torch
import torchvision

def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """

    x_norm = None


class NormalizationLayer(torch.nn.Module):
    """Class for normalization layer."""
    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)