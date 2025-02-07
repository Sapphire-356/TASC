from math import sqrt
import torch
from torch import baddbmm, bmm, softmax, cat, Tensor, empty, addmm
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn
from torch.nn import Linear as tLinear, Parameter, init
from torch.nn.functional import dropout
from copy import deepcopy


class Linear(tLinear):
    """
    LoRA wrapped Linear layer.
    """

    def __init__(self, in_features: int, out_features: int, *args, lora_r: int = 0, lora_alpha: float = 1.,
                 lora_dropout: float = 0., **kwargs):
        """
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout

        See torch.nn.Linear for other params
        """
        super().__init__(in_features, out_features, *args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.using_moving_average = False
        if lora_r:  # enable lora
            self.weight.requires_grad = False  # freeze main weights
            self.lora_a = Parameter(init.kaiming_uniform_(empty(lora_r, in_features), a=sqrt(5)))
            self.lora_b = Parameter(init.zeros_(empty(out_features, lora_r)))
            self.lora_dropout = lora_dropout
            self.lora_alpha = lora_alpha
            self._lora_scaling = lora_alpha / lora_r
            self.register_buffer(f'buffer_moving_avg_lora_a', self.lora_a.data.clone())
            self.register_buffer(f'buffer_moving_avg_lora_b', self.lora_b.data.clone())
            self.register_buffer(f'buffer_moving_avg_full_mat', torch.zeros((out_features, in_features)))

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        if self.lora_r:
            if self.training and self.lora_dropout:
                x = dropout(x, self.lora_dropout)
            actual_lora_a = self.lora_a if not self.using_moving_average else self.moving_avg_lora_a
            actual_lora_b = self.lora_b if not self.using_moving_average else self.moving_avg_lora_b
            a = x @ actual_lora_a.transpose(0, 1)
            return addmm(out.flatten(end_dim=-2), a.flatten(end_dim=-2), actual_lora_b.transpose(0, 1),
                         alpha=self._lora_scaling).view(out.shape)
        return out

    def merge_lora(self):
        """
        Transform LoRA linear to normal
        """
        if not self.lora_r:
            return
        self.weight.data += (self.lora_b @ self.lora_a) * self._lora_scaling
        self.weight.requires_grad = True
        self.lora_r = 0
        del self.lora_a, self.lora_b, self.lora_dropout, self.lora_alpha, self._lora_scaling

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if self.lora_r:
            return r + f', lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout}'
        return r
    
    def set_lora_moving_average(self, using_moving_average):
        self.using_moving_average = using_moving_average

    def lora_moving_average(self, lambda_last=None, lambda_current=None):
        self.moving_avg_lora_a.data = self.lora_a.data * lambda_current + self.moving_avg_lora_a.data * lambda_last
        self.moving_avg_lora_b.data = self.lora_b.data * lambda_current + self.moving_avg_lora_b.data * lambda_last
        tmp_mat = self.lora_a.T @ self.lora_b.T
        self.moving_avg_full_mat.data = tmp_mat.data * lambda_current + self.moving_avg_full_mat.data * lambda_last
    
    @property
    def moving_avg_lora_a(self):
        return getattr(self, f"buffer_moving_avg_lora_a")

    @property
    def moving_avg_lora_b(self):
        return getattr(self, f"buffer_moving_avg_lora_b")

    @property
    def moving_avg_full_mat(self):
        return getattr(self, f"buffer_moving_avg_full_mat")


def _update_lora(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        q_w, k_w, v_w = state_dict.pop(prefix + 'in_proj_weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'in_proj_bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
        state_dict[prefix + 'q_proj.bias'] = q_b
        state_dict[prefix + 'k_proj.bias'] = k_b
        state_dict[prefix + 'v_proj.bias'] = v_b
    elif prefix + 'qkv_proj.weight' in state_dict:  # transform packed projection
        q_w, k_w, v_w = state_dict.pop(prefix + 'qkv_proj.weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'qkv_proj.bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
        state_dict[prefix + 'q_proj.bias'] = q_b
        state_dict[prefix + 'k_proj.bias'] = k_b
        state_dict[prefix + 'v_proj.bias'] = v_b


def _update_packed(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        state_dict[prefix + 'qkv_proj.weight'] = state_dict.pop(prefix + 'in_proj_weight')
        state_dict[prefix + 'qkv_proj.bias'] = state_dict.pop(prefix + 'in_proj_bias')
    elif prefix + 'q_proj.weight' in state_dict:  # transform unpacked projection
        q_w = state_dict.pop(prefix + 'q_proj.weight')
        k_w = state_dict.pop(prefix + 'k_proj.weight')
        v_w = state_dict.pop(prefix + 'v_proj.weight')
        q_b = state_dict.pop(prefix + 'q_proj.bias')
        k_b = state_dict.pop(prefix + 'k_proj.bias')
        v_b = state_dict.pop(prefix + 'v_proj.bias')
        state_dict[prefix + 'qkv_proj.weight'] = cat([q_w, k_w, v_w])
        state_dict[prefix + 'qkv_proj.bias'] = cat([q_b, k_b, v_b])


class LORAMultiheadAttention(nn.Module):
    """
    LoRA wrapped Multi-Head Attention
    """

    def __init__(self, embed_dim, num_heads, dropout=0., lora_r: int = 0, lora_alpha: float = 1.,
                 lora_dropout: float = 0., batch_first=False):
        """
        :param embed_dim: the size of each embedding vector
        :param num_heads: number of heads
        :param dropout: attention dropout
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout
        """
        assert not embed_dim % num_heads, 'embed_dim must be divisible by num_heads'
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.lora_r = lora_r
        self._scale = 1 / sqrt(self.head_dim)

        if lora_r:
            self.q_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            # self.k_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.k_proj = nn.Linear(embed_dim, embed_dim)

            self.v_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self._register_load_state_dict_pre_hook(_update_lora)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
            self._register_load_state_dict_pre_hook(_update_packed)
        # self.o_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.batch_first = batch_first

    def forward(self, q, k, v, attn_mask, need_weights: bool = True):
        if self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        tgt_len, bsz, _ = q.shape
        src_len, _, _ = k.shape

        # do projection
        if self.lora_r:
            # print('gogogo')
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)
        else:  # self-attention
            q, k, v = self.qkv_proj(q).chunk(3, dim=-1)

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).permute(1, 2, 0)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if attn_mask is not None:
            a = attn_mask + self._scale * q @ k
        else:
            a = self._scale * q @ k

        a = softmax(a, dim=-1)
        if self.training and self.dropout:
            a = F.dropout(a, self.dropout)

        o = bmm(a, v).transpose(0, 1).contiguous().view(tgt_len * bsz, self.embed_dim)
        o = self.o_proj(o).view(tgt_len, bsz, -1)  # switch dimensions back
        if self.batch_first:
            o = o.transpose(0, 1)

        if need_weights:
            a = a.view(bsz, self.num_heads, tgt_len, src_len)
            a = a.sum(dim=1) / self.num_heads
            return o, a
        else:
            return o, None
        
    def set_lora_moving_average(self, using_moving_average=False):
        if self.lora_r:
            self.q_proj.set_lora_moving_average(using_moving_average)
            self.v_proj.set_lora_moving_average(using_moving_average)
            # self.o_proj.set_lora_moving_average(using_moving_average)
        else:
            self.qkv_proj.set_lora_moving_average(using_moving_average)


    def lora_moving_average(self, lambda_last=None, lambda_current=None):
        if self.lora_r:
            self.q_proj.lora_moving_average(lambda_last, lambda_current)
            self.v_proj.lora_moving_average(lambda_last, lambda_current)
            # self.o_proj.lora_moving_average(lambda_last, lambda_current)
        else:
            self.qkv_proj.lora_moving_average(lambda_last, lambda_current)


def LoRA_ViT(old_model, lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.,
             lora_other_linear: bool = False, batch_first=False):
    """
    replace all multihead attention with LoRA multihead attention, and copy the weights to the new model
    :param old_model:
    :return:
    """
    model = deepcopy(old_model)

    # for name, module in old_model.named_modules():
    #     if isinstance(module, nn.MultiheadAttention):
    #         setattr(model, name, LORAMultiheadAttention(module.embed_dim, module.num_heads, module.dropout,
    #                                                     lora_r=lora_r, lora_alpha=lora_alpha,
    #                                                     lora_dropout=lora_dropout))
    #         getattr(model, name).load_state_dict(module.state_dict(), strict=False)
    #
    #
    def change_module(module, prefix):
        actual_prefix = prefix + '.' if prefix else ''
        for name, child in module.named_children():
            if isinstance(child, nn.MultiheadAttention):
                setattr(module, name, LORAMultiheadAttention(child.embed_dim, child.num_heads, child.dropout,
                                                             lora_r=lora_r, lora_alpha=lora_alpha,
                                                             lora_dropout=lora_dropout, batch_first=batch_first))
            elif lora_other_linear:
                if isinstance(child, nn.Linear):
                    setattr(module, name, Linear(child.in_features, child.out_features, lora_r=lora_r,
                                                 lora_alpha=lora_alpha, lora_dropout=lora_dropout))
                else:
                    change_module(child, actual_prefix + name)
            else:
                change_module(child, actual_prefix + name)

    change_module(model, '')
    model.load_state_dict(old_model.state_dict(), strict=False)
    return model


def lora_moving_average(model, lambda_last=None, lambda_current=None):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.lora_moving_average(lambda_last=lambda_last, lambda_current=lambda_current)
            else:
                change_module(child)

    #
    change_module(model)


def set_lora_moving_average(model, using_moving_average=False):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.set_lora_moving_average(using_moving_average)
            else:
                change_module(child)

    #
    change_module(model)