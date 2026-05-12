import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []

        # Convolution Layer
        layers += [nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]

        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(DeconvBlock, self).__init__()
        layers = []

        # Transpose Convolution Layer
        layers += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]

        # Non-Linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'tanh':
            layers += [nn.Tanh()]

        self.deconv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.deconv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=8):
        super(Generator, self).__init__()
        self.downsample_1 = ConvBlock(2 + z_dim, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu')
        self.downsample_2 = ConvBlock(32, 64, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_3 = ConvBlock(64, 128, k=4, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_4 = ConvBlock(128, 256, k=4, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_5 = ConvBlock(256, 256, k=4, s=1, p=1, norm=True, non_linear='leaky_relu')

        # Need concatenation when upsampling, see foward function for details
        self.upsample_1 = DeconvBlock(256, 256, k=4, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.upsample_2 = DeconvBlock(512, 128, k=4, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.upsample_3 = DeconvBlock(256, 64, k=4, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.upsample_4 = DeconvBlock(128, 32, k=4, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.upsample_5 = DeconvBlock(64, 1, k=4, s=2, p=1, norm=False, non_linear='tanh')

    def forward(self, x, seismic, z):
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, seismic, z], dim=1)

        down_1 = self.downsample_1(x_with_z)
        down_2 = self.downsample_2(down_1)
        down_3 = self.downsample_3(down_2)
        down_4 = self.downsample_4(down_3)
        down_5 = self.downsample_5(down_4)

        up_1 = self.upsample_1(down_5)  # 1 4 4
        up_2 = self.upsample_2(torch.cat([up_1, down_4], dim=1))
        up_3 = self.upsample_3(torch.cat([up_2, down_3], dim=1))
        up_4 = self.upsample_4(torch.cat([up_3, down_2], dim=1))
        out = self.upsample_5(torch.cat([up_4, down_1], dim=1))

        return out


class Discriminator(nn.Module):
    def __init__(self, ndim):
        super(Discriminator, self).__init__()
        self.ndim = ndim

        self.d_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=0, count_include_pad=False),
            ConvBlock(self.ndim, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
            ConvBlock(32, 64, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'),
            ConvBlock(64, 128, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'),
            ConvBlock(128, 1, k=3, s=1, p=1, norm=False, non_linear=None),
        )

        self.d_2 = nn.Sequential(
            ConvBlock(self.ndim, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
            ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'),
            ConvBlock(128, 256, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'),
            ConvBlock(256, 1, k=3, s=1, p=1, norm=False, non_linear=None),
        )

    def forward(self, x):
        out_1 = self.d_1(x)
        out_2 = self.d_2(x)
        return (out_1, out_2)


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.Conv2d(in_dim, out_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )

        self.short_cut = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


class Encoder(nn.Module):
    def __init__(self, z_dim=8):
        super(Encoder, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.res_blocks = nn.Sequential(
            ResBlock(32, 64),
            ResBlock(64, 128),
            ResBlock(128, 256),
        )

        self.pool_block = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(kernel_size=(5, 6), stride=1, padding=0),  # 5*6->1*1
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
        )

        # Return mu and logvar for reparameterization trick
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x):
        out = self.conv(x)
        out = self.res_blocks(out)
        out = self.pool_block(out)
        out = out.view(x.size(0), -1)

        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)

        return (mu, log_var)
