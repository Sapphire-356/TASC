
import clip

CLIP_MODELS = clip.available_models()  
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
DINOv2_MODELS = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']
backbone_names = ['resnet50'] + CLIP_MODELS + DINOv2_MODELS


def build_backbone_local(name):
    """
    by xxx (20230719) (modified from uniood)

    build the backbone for feature exatraction

    args:
        repo_dir: The directory of github repositories, such as CLIP and dinov2. 
                This is done because the torch.hub.load() still requires 
                the hubconf.py file when loading the model with source='local'.
    """
    import os
    repo_dir = os.path.abspath(os.getcwd())
    if name == '':
        raise NotImplementedError()
    # elif name == 'resnet50':
    #     return ResBase(option=name)
    elif name in CLIP_MODELS:
        model, _ = clip.load(name, download_root=os.path.join(repo_dir, "data/models/clip"))
    elif name in DINOv2_MODELS:
        import torch, os
        model = torch.hub.load(os.path.join(repo_dir, 'dinov2'), name, source='local')
    else:
        raise RuntimeError(f"Model {name} not found; available models = {backbone_names}")
    
    return model.float()