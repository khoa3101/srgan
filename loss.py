import torch
import torch.nn as nn
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1

class GeneratorLoss(nn.Module):
    def __init__(self, mode='vgg19'):
        super(GeneratorLoss, self).__init__()

        if mode == 'vgg19':
            vgg = models.vgg19_bn(pretrained=True)
            self.model = nn.Sequential(*(list(vgg.children())[:51])).eval()
        elif mode == 'inception1':
            incept = InceptionResnetV1(pretrained='vggface2')
            self.model = nn.Sequential(*(list(incept.children())[:13])).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.mse = nn.MSELoss()
        self.tv = TVLoss()

    
    def forward(self, sr, hr, fake_labels):
        adv_loss = torch.mean(1 - fake_labels)
        content_loss = self.mse(hr, sr)
        feature_loss = self.mse(self.model(hr), self.model(sr))
        tv_loss = self.tv(sr)

        return content_loss + 1e-3*adv_loss + 6e-3*feature_loss + 2e-8*tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]