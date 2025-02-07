import torch.nn as nn
import torch.nn.functional as F



class CLS(nn.Module):
    def __init__(self, num_classes, input_size=2048, bias=False, cfg=None):
        super(CLS, self).__init__()
        self.C_K = nn.Linear(input_size, num_classes, bias=bias)
        self.num_classes = num_classes
        self.cfg = cfg
        self.input_size = input_size

    def forward(self, x):
        logits_K = self.C_K(x)
        return logits_K
    
    def weight_norm(self):
        w = self.C_K.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.C_K.weight.data = w.div(norm.expand_as(w))



class ProtoCLS(nn.Module):
    """
    (from uniood)
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05, cfg=None):
        super(ProtoCLS, self).__init__()
        self.num_classes = out_dim
        self.cfg = cfg
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp 
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))



class ResClassifier_MME(nn.Module):
    '''
    the same as DANCE and OVANet
    '''
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)



class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=None):
        super(MLP, self).__init__()
        if h_dim == None: h_dim =in_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, h_dim, bias=True),
                                nn.ReLU(), 
                                nn.Linear(h_dim, out_dim, bias=True),
                                )

    def forward(self, x):
        output = self.mlp(x)
        return output
    


class Projection(nn.Module):
    """
    a projection head
    """
    def __init__(self, in_dim, hidden_mlp=2048, feat_dim=256):
        super(Projection, self).__init__()
        self.projection_head = nn.Sequential(
                               nn.Linear(in_dim, hidden_mlp),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_mlp, feat_dim))
        self.output_dim = feat_dim

    def forward(self, x):
        return self.projection_head(x)
    


class ProjectCLS(nn.Module):
    """
    (from UniOT https://github.com/changwxx/UniOT-for-UniDA)
    a classifier made up of projection head and prototype-based classifier
    """
    def __init__(self, in_dim, out_dim, hidden_mlp=2048, feat_dim=256, temp=0.05):
        super(ProjectCLS, self).__init__()
        self.projection_head = nn.Sequential(
                            nn.Linear(in_dim, hidden_mlp),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_mlp, feat_dim))
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls
    


class EmbeddingBN(nn.Module):
    '''
    from https://github.com/ispc-lab/GLC
    '''

    def __init__(self, feature_dim, embed_dim=256, type="bn"):
        super(EmbeddingBN, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(self.init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x
    
    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)


class ClassifierWN(nn.Module):
    '''
    from https://github.com/ispc-lab/GLC
    '''

    def __init__(self, embed_dim, class_num, type="wn"):
        super(ClassifierWN, self).__init__()
        
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            self.fc.apply(self.init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(self.init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)