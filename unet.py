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
        channels: int = 1,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        num_classes: int = 10,
        output_attention: bool = False,
    ) -> None:
        super().__init__()

        # determine dimensions

        channels = 1
        self.channels = channels
        # if you want to do self conditioning uncomment this
        input_channels = channels
        self.output_attention = output_attention

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)
        dims = [init_dim, *(dim * m for m in dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        self.protein_emb = nn.Linear(1280,400)
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        self.cross_attn = EfficientAttention(
            dim=100,
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )
        self.norm_to_cross = nn.LayerNorm(dim * 4)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):
        # [batch_size, 1,  4,200]
        x = self.init_conv(x)
        # [batch_size, 200, 4,200] ？？？？
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        t_cross = t_start.clone()
        # [batch_size, 800] ？？？？
        # if classes is not None:
        #     t_start += self.label_emb(classes)
        #     #[batch,800]??
        #     t_mid += self.label_emb(classes)
        #     t_end += self.label_emb(classes)
        #     t_cross += self.label_emb(classes)
        if classes is not None:
            t_start += self.protein_emb(classes)
            #[batch,800]??
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
        #[batch_size, 800, 4, 200]
        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)
        # [batch_size, 800, 4, 200]
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)

            x = upsample(x)
        # [batch_size, 200, 4, 200]    
        x = torch.cat((x, r), dim=1)
        # [batch_size, 400, 4, 200]
        x = self.final_res_block(x, t_end)
        x = self.final_conv(x)
        x_reshaped = x.reshape(-1, 4, 100)
        t_cross_reshaped = t_cross.reshape(-1, 4, 100)
        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, 400)).reshape(-1, 4, 100),
            context=t_cross_reshaped,
        )  
        crossattention_out = crossattention_out.view(-1, 1, 4, 100)
        x = x + crossattention_out
        if self.output_attention:
            return x, crossattention_out
        return x
    
# x = torch.randn(16, 1, 4, 100)
# time = torch.randn(16)
# classes = torch.randn(16, 1280)  # Example protein embeddings
# unet = UNet(
#     dim=100,
#     channels=1,
#     dim_mults=(1, 2, 4),
#     resnet_block_groups=4,
# )
# output = unet(x, time, classes)
# print(output.shape)  # Should be (16, 1, 4, 200)
# print(output)

class UNetV2(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        dim_mults: tuple = (1, 2, 4),
        channels: int = 1,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        num_classes: int = 10,
        output_attention: bool = False,
    ) -> None:
        super().__init__()

        # determine dimensions

        channels = 1
        self.channels = channels
        # if you want to do self conditioning uncomment this
        input_channels = channels
        self.output_attention = output_attention

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)
        dims = [init_dim, *(dim * m for m in dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        self.protein_emb = nn.Linear(2560,400)
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        self.cross_attn = EfficientAttention(
            dim=100,
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )
        self.norm_to_cross = nn.LayerNorm(dim * 4)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):
        # [batch_size, 1, 4, 200]
        # [batch_size, 1, 16, 16]
        x = self.init_conv(x)
        # [batch_size, 200, 4, 200]
        # [batch_size, 200, 16, 16] 
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        t_cross = t_start.clone()
        # [batch_size, 800] ？？？？
        # if classes is not None:
        #     t_start += self.label_emb(classes)
        #     #[batch,800]??
        #     t_mid += self.label_emb(classes)
        #     t_end += self.label_emb(classes)
        #     t_cross += self.label_emb(classes)
        if classes is not None:
            t_start += self.protein_emb(classes)
            #不同的维度的变化应该是[batch,16*16] -> [batch,256] 应该是对应的维度
            #[batch,800]??
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
        # [batch_size, 800, 4, 200]
        # [batch_size, 800, 16, 16]
        x = self.mid_block1(x, t_mid)

        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)
        # [batch_size, 800, 4, 200]
        # [batch_size, 800, 16, 16]
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)

            x = upsample(x)
        # [batch_size, 200, 16, 16]
        # [batch_size, 200, 4, 200]    
        x = torch.cat((x, r), dim=1)
        # [batch_size, 400, 4, 200]
        x = self.final_res_block(x, t_end)
        x = self.final_conv(x)
        x_reshaped = x.reshape(-1, 4, 100)
        t_cross_reshaped = t_cross.reshape(-1, 4, 100)
        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, 400)).reshape(-1, 4, 100),
            context=t_cross_reshaped,
        )  # (-1,1, 4, 200)
        crossattention_out = crossattention_out.view(-1, 1, 4, 100)
        x = x + crossattention_out
        if self.output_attention:
            return x, crossattention_out
        return x




class UNetV3(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        dim_mults: tuple = (1, 2, 4),
        # MODIFIED: Input channels now default to 4 for the latent space shape [B, 4, 16, 16]
        channels: int = 4,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        protein_emb_dim: int = 1280, # Dimension of the input protein embedding
    ) -> None:
        super().__init__()

        # --- Dimension Setup ---
        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        # MODIFIED: init_conv now accepts `input_channels` (e.g., 4)
        # self.init_conv = nn.Conv2d(input_channels, init_dim, (3, 3), padding=1)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (3, 3), padding=1)
        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # --- Time and Protein/Class Embedding ---
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

        # --- Architecture Layers (Downsampling) ---
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

        # --- Bottleneck ---
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # --- Architecture Layers (Upsampling) ---
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
        # --- Final Processing Layers ---
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
        """  
        # ADDED: A proper final cross-attention block.
        # This assumes your 'Attention' class can accept a 'context_dim' argument for cross-attention.
        self.final_attn = Residual(PreNorm(dim, Attention(dim, context_dim=time_dim)))

        # MODIFIED: final_conv now outputs `channels` (e.g., 4) to match the input shape.
        self.final_conv = nn.Conv2d(dim, self.channels, 1)
        """
    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):
        # Input x: [B, 4, 8, 8]

        # Initial convolution
        x = self.init_conv(x)  # -> [B, 100, 8, 8]
        r = x.clone()

        # Prepare time and protein condition embeddings
        t_start = self.time_mlp(time)                # -> [B, 256]
        t_mid = t_start.clone()                      # -> [B, 256]
        t_end = t_start.clone()                      # -> [B, 256]
        t_cross = t_start.clone()                    # -> [B, 256]
        if classes is not None:
            t_start += self.protein_emb(classes)     # -> [B, 256]
            t_mid += self.protein_emb(classes)       # -> [B, 256]
            t_end += self.protein_emb(classes)       # -> [B, 256]
            t_cross += self.protein_emb(classes)     # -> [B, 256]

        h = []

        # Downsampling path
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)
            h.append(x)
            x = block2(x, t_start)
            x = attn(x)
            h.append(x)
            # x = crossattn(x, context=t)
            x = downsample(x)
        # After downsampling loop: x -> [B, 400, 2, 2]

        # Bottleneck
        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid) # -> [B, 400, 2, 2]
        # Upsampling path
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_end)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_end)
            x = attn(x)
            # x = crossattn(x, context=t)
            x = upsample(x)
        # After upsampling loop: x -> [B, 100, 4, 4]
       
        # Final blocks
        x = torch.cat((x, r), dim=1)            # -> [B, 200, 8, 8]
        x = self.final_res_block(x, t_end)         # -> [B, 100, 16, 16]
        x = self.final_conv(x)   #-> [B, 4, 8, 8]
        x_reshaped = x.reshape(-1, 4, 64)
        t_cross_reshaped = t_cross.reshape(-1, 4, 64)
        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, 256)).reshape(-1, 4, 64),
            context=t_cross_reshaped,
        )  # (-1,1, 4, 64)
        crossattention_out = crossattention_out.view(-1, 4, 8, 8)
        x = x + crossattention_out
        return x
# x = torch.randn(16, 4, 8, 8)
# time = torch.randn(16)
# classes = torch.randn(16, 1280)  # Example protein embeddings
# unet = UNetV3(
#     dim=100,
#     channels=4,
#     dim_mults=(1, 2, 4),
#     resnet_block_groups=4,
# )
# output = unet(x, time, classes)
# print(output.shape)  # Should be [16, 4, 8, 8]