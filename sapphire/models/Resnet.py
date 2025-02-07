
from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
import os


class ResNet(nn.Module):
    def __init__(self, option='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        self.dim = 2048
        # if option == 'resnet18':
        #     model_ft = models.resnet18(weights='IMAGENET1K_V1')
        #     self.dim = 512
        # if option == 'resnet34':
        #     model_ft = models.resnet34(weights='IMAGENET1K_V1')
        #     self.dim = 512
        # if option == 'resnet50':
        #     model_ft = models.resnet50(weights='IMAGENET1K_V1')
        # if option == 'resnet101':
        #     model_ft = models.resnet101(weights='IMAGENET1K_V1')
        # if option == 'resnet152':
        #     model_ft = models.resnet152(weights='IMAGENET1K_V1')
        
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pretrained)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pretrained)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pretrained)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pretrained)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pretrained)

        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        if x.ndim == 1: x = x.unsqueeze(0)
        return x



class BaseFeatureExtractor(nn.Module):
    '''
    copy from UniOT https://github.com/changwxx/UniOT-for-UniDA
    From https://github.com/thuml/Universal-Domain-Adaptation
    a base class for feature extractor
    '''
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)



class ResNet50Fc(BaseFeatureExtractor):
    """
    copy from UniOT https://github.com/changwxx/UniOT-for-UniDA
    modefied from https://github.com/thuml/Universal-Domain-Adaptation
    implement ResNet50 as backbone, but the last fc layer is removed
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                model_resnet = models.resnet50(pretrained=False)
                model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        mod = list(model_resnet.children())
        mod.pop()
        self.feature_extractor = nn.Sequential(*mod)
        self.dim = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x