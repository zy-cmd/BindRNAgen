import torch.nn as nn

class DecoderBlock(nn.Module):
    """
    A 2D deconvolutional block used in the Decoder, analogous to a residual block.
    """
    def __init__(self, in_channels, out_channels, stride=(1, 2), output_padding=(0,1)):
        super(DecoderBlock, self).__init__()
        self.deconv_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=(1,1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        return self.final_relu(self.deconv_path(x) + self.shortcut(x))

class Decoder2D(nn.Module):
    """
    The Decoder part of the VAE.
    Takes a latent vector z and reconstructs the 2D RNA sequence representation.
    """
    def __init__(self, latent_channels):
        super(Decoder2D, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.upsample = nn.Upsample(size=(8, 13), mode='bilinear', align_corners=False)
        self.deconv_block1 = DecoderBlock(128, 64, stride=(1, 2), output_padding=(0, 1))
        self.deconv_block2 = DecoderBlock(64, 32, stride=(1, 2), output_padding=(0, 0))
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 0)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
        )

    def forward(self, z):
        z = self.initial_conv(z)
        z = self.upsample(z)
        z = self.deconv_block1(z)
        z = self.deconv_block2(z)
        z = self.final_deconv(z)
        z = z.squeeze(1)
        return z


class ModifiedDecoder2D(nn.Module):
 
    def __init__(self, latent_channels):
        super(ModifiedDecoder2D, self).__init__()


        self.init_conv = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )


        self.decode_stage1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(1, 6), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        self.decode_stage2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )


        self.decode_stage3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )


        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid() # 通常在最后使用Tanh或Sigmoid将输出归一化到[-1, 1]或[0, 1]
        )

    def forward(self, z):
        x = self.init_conv(z)         # -> (16, 128, 8, 8)
        x = self.decode_stage1(x)     # -> (16, 128, 16, 26)
        x = self.decode_stage2(x)     # -> (16, 64, 8, 51)
        x = self.decode_stage3(x)     # -> (16, 32, 4, 101)
        x = self.final_conv(x)        # -> (16, 1, 4, 101)
        
        return x.squeeze(1)           # -> (16, 4, 101)
