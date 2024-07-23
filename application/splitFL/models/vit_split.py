'''
Standard ViT split model adapt from 

https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

'''

import torch
import torch.nn.functional as F
from torch import nn
from models.split_models import BaseSFLModel


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class clstoken_add(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    def forward(self, x):
        b, n, _ = x.shape # b, 197
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        return torch.cat((cls_tokens, x), dim=1)

class posemd_add(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
    def forward(self, x):
        b, n, _ = x.shape # b, 197
        return x + self.pos_embedding[:, :n]

class ViewLayer(nn.Module):
    def __init__(self, pool = 'cls'):
        super().__init__()
        self.pool = pool
    
    def forward(self, x):
        if self.pool == 'mean':
            return x.mean(dim = 1)
        else:
            return x[:, 0]

class FeedForward_res(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x) + x

class Attention_res(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        qkv = self.to_qkv(self.norm(x)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) + x

def create_tranformer_list(dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
    transformer_list = []

    for _ in range(depth):
        transformer_list.append(Attention_res(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        transformer_list.append(FeedForward_res(dim, mlp_dim, dropout = dropout))
    transformer_list.append(nn.LayerNorm(dim))

    return transformer_list

class Model(BaseSFLModel):
    def __init__(self, *, image_size = 32, patch_size = 4, num_classes = 10, dim = 512, depth = 6, heads = 8, mlp_dim = 512, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        self.num_channels = image_size
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        model_list = [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                        nn.LayerNorm(patch_dim),
                        nn.Linear(patch_dim, dim),
                        nn.LayerNorm(dim), # to patch embedding,
                        clstoken_add(dim), # concat cls token
                        posemd_add(num_patches, dim = dim), # add pos embedding
                        nn.Dropout(emb_dropout)]
        
        model_list.extend(create_tranformer_list(dim, depth, heads, dim_head, mlp_dim, dropout))
                        
        model_list.extend([ViewLayer(pool),
                        nn.Identity(),
                        nn.Linear(dim, num_classes)])

        self.model = nn.Sequential(*model_list)

        self.backbone_output_dim = dim

        self.split()

        self.initialize_weights()
    
    def check_if_module_is_layer(self, module):
        '''
        Override this method if model includes modules other than Conv2d and Linear (i.e. BasicBlock)
        '''
        valid_layer_list = ["Linear", "Attention_res", "FeedForward_res"]
        for valid_layer in valid_layer_list:
            if valid_layer in str(module):
                return True
        return False