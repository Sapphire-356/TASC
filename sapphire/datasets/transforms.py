
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, RandomResizedCrop,
    RandomHorizontalFlip
)

from mmcls.datasets import PIPELINES
from mmcls.datasets.pipelines import Compose as MMCompose
from .mmcls_transforms import rand_range_aug


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

SIZE = (224, 224)
# MEAN and STD
IMAGENET_PIXEL_MEAN = [0.485, 0.456, 0.406]
IMAGENET_PIXEL_STD = [0.229, 0.224, 0.225]
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

# default is 'bilinear', here we set it 'bicubic' following uniood 
INTERPOLATION = INTERPOLATION_MODES["bicubic"]
# Random crop
CROP_PADDING = 0
# Random resized crop
RRCROP_SCALE = (0.08, 1.0)



data_transforms = {
        'weak': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD)
        ]),
    }





def get_train_pipelines(mean, std):

    img_norm_cfg = dict(
        mean=[i*255.0 for i in mean],
        std=[i*255.0 for i in std],
        to_rgb=True)

    train_pipelines = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=256, backend='pillow'),
        dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
        dict(type='RandomCrop', size=224),
        rand_range_aug,
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=['gt_label']),
        dict(type='Collect', keys=['img', 'gt_label'])
    ]
    
    return train_pipelines



def get_data_transforms(transforms_type: str, 
                        backbone_name=None,
                        size=SIZE,
                        interpolation=INTERPOLATION,
                        pixel_mean=IMAGENET_PIXEL_MEAN,
                        pixel_std=IMAGENET_PIXEL_STD,
                        crop_padding=CROP_PADDING,
                        rrcrop_scale=RRCROP_SCALE):
    '''
    args:
        crop_padding: Used in Randomcrop. Default is no padding. If `crop_padding`
            is not 0, then padding `crop_padding` pixels on the borders of image.
            The `padding_mode` is 'constant' by default. The number padded is 0
            by default.
    
    '''
    
    if 'clip' in transforms_type:
        transforms_type = transforms_type.split('-')[1]
        pixel_mean = CLIP_PIXEL_MEAN
        pixel_std = CLIP_PIXEL_STD

        if backbone_name == 'RN50x4':
            size = (288, 288)
        elif backbone_name == 'RN50x16':
            size = (384, 384)
        elif backbone_name == 'ViT-L/14@336px':
            size = (336, 336)

        normalize = transforms.Normalize(mean=pixel_mean, std=pixel_std)

        clip_data_transforms = {
            "none":  # equal to data_transforms['val']
                Compose([
                    Resize(size=max(size), interpolation=interpolation),
                    CenterCrop(size=size),
                    ToTensor(),
                    normalize,
                ]),
            "flip":  
                Compose([
                    Resize(size=max(size), interpolation=interpolation),
                    CenterCrop(size=size),  # center crop
                    RandomHorizontalFlip(p=1.0),  # the probability of flip is 1.0 
                    ToTensor(),
                    normalize,
                ]),
            "randomcrop":  # equal to data_transforms['weak']
                Compose([
                    Resize(size=max(size), interpolation=interpolation),
                    RandomCrop(size=size, padding=crop_padding),  
                    RandomHorizontalFlip(p=0.5),
                    ToTensor(),
                    normalize,
                ]),
            "randomresizedcrop":
                Compose([  # the speacial Crop method
                    RandomResizedCrop(size=size, scale=rrcrop_scale, interpolation=interpolation),
                    RandomHorizontalFlip(p=0.5),
                    ToTensor(),
                    normalize,
                ]),   
            "strong_1":
                Compose([
                    transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0), interpolation=interpolation),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
            "strong_2": MMCompose(get_train_pipelines(pixel_mean, pixel_std)),
        }

        return clip_data_transforms[transforms_type]

    else:
        return data_transforms[transforms_type]
    
