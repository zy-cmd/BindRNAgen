import itertools
from functools import partial
from typing import Optional
import torch.nn as nn
import torch
from memory_efficient_attention_pytorch import Attention as EfficientAttention

from layers import *

class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        dim_mults: tuple = (1, 2, 4),
        channels: int = 4,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        protein_emb_dim: int = 1280, 
    ) -> None:
        super().__init__()

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (3, 3), padding=1)
        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = 256
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.protein_emb = nn.Linear(protein_emb_dim, time_dim)

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    # Residual(MultiHeadCrossAttention(dim_q = dim_in, dim_kv = 400,hidden_dim = 64, num_heads=4)),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    # Residual(MultiHeadCrossAttention(dim_q = dim_out, dim_kv = 400,hidden_dim = 64, num_heads=4)),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 4, 1)
        self.cross_attn = EfficientAttention(
            dim=64,
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )
        self.norm_to_cross = nn.LayerNorm(256)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):

        x = self.init_conv(x)  
        r = x.clone()

        t_start = self.time_mlp(time)               
        t_mid = t_start.clone()                   
        t_end = t_start.clone()                     
        t_cross = t_start.clone()                   
        if classes is not None:
            t_start += self.protein_emb(classes)   
            t_mid += self.protein_emb(classes)      
            t_end += self.protein_emb(classes)      
            t_cross += self.protein_emb(classes)   

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)
            h.append(x)
            x = block2(x, t_start)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid) 
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_end)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_end)
            x = attn(x)
            x = upsample(x)
       
        x = torch.cat((x, r), dim=1)           
        x = self.final_res_block(x, t_end)       
        x = self.final_conv(x)  
        x_reshaped = x.reshape(-1, 4, 64)
        t_cross_reshaped = t_cross.reshape(-1, 4, 64)
        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, 256)).reshape(-1, 4, 64),
            context=t_cross_reshaped,
        ) 
        crossattention_out = crossattention_out.view(-1, 4, 8, 8)
        x = x + crossattention_out
        return x
