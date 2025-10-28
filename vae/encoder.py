import torch.nn as nn

class ResidualBlock2D(nn.Module):
    """
    A 2D residual block used in the Encoder.
    Includes two convolutional paths and a shortcut connection.
    """
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResidualBlock2D, self).__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        return self.final_relu(self.conv_path(x) + self.shortcut(x))

class Encoder2D(nn.Module):
    """
    The Encoder part of the VAE.
    Takes a 2D RNA sequence representation and encodes it into latent space variables mu and log_var.
    """
    def __init__(self, latent_channels):
        super(Encoder2D, self).__init__()
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.res_block1 = ResidualBlock2D(32, 64, stride=(1, 2))
        self.res_block2 = ResidualBlock2D(64, 128, stride=(1, 2))
        self.final_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)), nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.conv_log_var = nn.Conv2d(128, latent_channels, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stage1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.final_conv(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        return mu, log_var





class ModifiedEncoder2D(nn.Module):

    def __init__(self, latent_channels):
        super(ModifiedEncoder2D, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=(1, 6), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.conv_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.conv_log_var = nn.Conv2d(128, latent_channels, kernel_size=1)

    def forward(self, x):
        # 假设输入 x 的形状是 (16, 4, 101)
        x = x.unsqueeze(1)      # -> (16, 1, 4, 101)
        x = self.stage1(x)      # -> (16, 32, 4, 101)
        x = self.stage2(x)      # -> (16, 64, 8, 51)
        x = self.stage3(x)      # -> (16, 128, 16, 26)
        x = self.stage4(x)      # -> (16, 128, 8, 8)
        mu = self.conv_mu(x)          # -> (16, latent_channels, 8, 8)
        log_var = self.conv_log_var(x)  # -> (16, latent_channels, 8, 8)
        
        return mu, log_var
    
