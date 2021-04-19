import numpy as np 
import torch
import torchvision

def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """

    # view的作用类似reshape
    # 如果不关心底层数据是否使用了新的内存，则使用reshape方法替代view更方便
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        # transpose把y的0维和1维换位置
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


class MyTripletLossFunc(torch.autograd.Function):

    @staticmethod
    def random_triplets(features):
        size = features.shape[0]

        triplets = []
        labels = list(range(int(size / 2))) + list(range(int(size / 2)))
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
        return triplets

    @staticmethod
    def forward(ctx, features):

        distances = pairwise_distances(features).cpu().numpy()
        triplets = MyTripletLossFunc.random_triplets(features)
    
        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for i, j, k in triplets:
            w = 1.0
            triplet_count += w
            loss += w * np.log(1 + np.exp(distances[i, j] - distances[i, k]))
            if distances[i, j] < distances[i, k]:
                correct_count += 1
        
        loss /= triplet_count

        triplet_count = torch.autograd.Variable(torch.FloatTensor([triplet_count]))
        distances = torch.autograd.Variable(torch.FloatTensor(distances))
        triplets = torch.autograd.Variable(torch.FloatTensor(triplets))
        ctx.save_for_backward(features, triplets, triplet_count, distances)

        return torch.FloatTensor((loss,))

    @staticmethod
    def backward(ctx, grad_output):
        features, triplets, triplet_count, distances = ctx.saved_tensors
        features_np = features.cpu().numpy()
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features.cpu().numpy()

        for i, j, k in triplets:
            i = int(i.item())
            j = int(j.item())
            k = int(k.item())
            w = 1.0
            f = 1.0 - 1.0 / (1.0 + np.exp(distances[i, j] - distances[i, k]))
            grad_features_np[i, :] += (w * f * (features_np[i, :] - features_np[j, :]) / triplet_count).numpy()
            grad_features_np[j, :] += (w * f * (features_np[j, :] - features_np[i, :]) / triplet_count).numpy()
            grad_features_np[i, :] += (-w * f * (features_np[i, :] - features_np[k, :]) / triplet_count).numpy()
            grad_features_np[k, :] += (-w * f * (features_np[k, :] - features_np[i, :]) / triplet_count).numpy()
        
        for i in range(features_np.shape[0]):
            grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
        grad_features *= float(grad_output.data[0])
        return grad_features

class MyTripletLossFunc_(torch.autograd.Function):

    def __init__(self, triplets):
        super(MyTripletLossFunc, self).__init__()
        self.triplets = triplets
        self.triplet_count = len(triplets)
    
    @staticmethod
    def forward(self, features):
        self.save_for_backward(features)

        print('------------', type(features), '\n')
        print(features)
        print('-----------------------', type(self.triplets), '\n')
        print(self.triplets)

        self.distances = pairwise_distances(features).cpu().numpy()

        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for i, j, k in self.triplets:
            w = 1.0
            triplet_count += w
            loss += w * np.log(1 + np.exp(self.distances[i, j] - self.distances[i, k]))
            if self.distances[i, j] < self.distances[i, k]:
                correct_count += 1
        
        loss /= triplet_count
        return torch.FloatTensor((loss, ))

    @staticmethod    
    def backward(self, grad_output):
        features, = self.save_tensors
        features_np = features.cpu().numpy()
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features.cup().numpy()

        for i, j, k in self.triplets:
            w = 1.0
            f = 1.0 - 1.0 / (1.0 + np.exp(self.distances[i, j] - self.distances[i, k]))
            grad_features_np[i, :] += w * f * (features_np[i, :] - features_np[j, :]) / self.triplet_count
            grad_features_np[j, :] += w * f * (features_np[j, :] - features_np[i, :]) / self.triplet_count
            grad_features_np[i, :] += -w * f * (features_np[i, :] - features_np[k, :]) / self.triplet_count
            grad_features_np[k, :] += -w * f * (features_np[k, :] - features_np[i, :]) / self.triplet_count

        for i in range(features_np.shape[0]):
            grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
        grad_features *= float(grad_features.data[0])
        return grad_features


class TripletLoss(torch.nn.Module):
    """Class for the triplet loss."""
    def __init__(self, pre_layer=None):
        super(TripletLoss, self).__init__()
        self.pre_layer = pre_layer

    def forward(self, x):
        if self.pre_layer is not None:
            x = self.pre_layer(x)
        loss = MyTripletLossFunc.apply(x)
        return loss


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