'''
Simple ViT split model adapt from 

https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

'''

import torch
import torch.nn.functional as F
from torch import nn
from models.split_models import BaseSFLModel


import torch
from torch import nn

from einops import rearrange
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

class posemd_add(nn.Module):
    def __init__(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
        super().__init__()
        self.pos_embedding = posemb_sincos_2d(h, w, dim = dim) 

    def forward(self, x):
        return x + self.pos_embedding.to(x.device, dtype=x.dtype)

class ViewLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean(dim = 1)

class FeedForward_res(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x) + x

class Attention_res(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) + x

def create_tranformer_list(dim, depth, heads, dim_head, mlp_dim):
    transformer_list = []

    for _ in range(depth):
        transformer_list.append(Attention_res(dim, heads = heads, dim_head = dim_head))
        transformer_list.append(FeedForward_res(dim, mlp_dim))
    transformer_list.append(nn.LayerNorm(dim))

    return transformer_list

class Model(BaseSFLModel):
    def __init__(self, *, image_size = 32, patch_size = 4, num_classes = 10, dim = 512, depth = 6, heads = 8, mlp_dim = 512, channels = 3, dim_head = 64):
        super().__init__()
        self.num_channels = image_size
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.pool = "mean"

        model_list = [Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
                        nn.LayerNorm(patch_dim),
                        nn.Linear(patch_dim, dim),
                        nn.LayerNorm(dim), # to patch embedding,
                        posemd_add(image_height // patch_height, image_width // patch_width, dim = dim)] # add pos embedding
        
        model_list.extend(create_tranformer_list(dim, depth, heads, dim_head, mlp_dim))
                        
        model_list.extend([ViewLayer(),
                        nn.Identity(),
                        nn.Linear(dim, num_classes)])

        self.model = nn.Sequential(*model_list)

        self.backbone_output_dim = dim

        self.split()

        self.initialize_weights()

    def forward(self, img):
        return self.model(img)

    def check_if_module_is_layer(self, module):
        '''
        Override this method if model includes modules other than Conv2d and Linear (i.e. BasicBlock)
        '''
        valid_layer_list = ["Linear", "Attention_res", "FeedForward_res"]
        for valid_layer in valid_layer_list:
            if valid_layer in str(module):
                return True
        return False
