import torch
from torch import nn
import math

hs_channels = 31
ms_channels = 3

"""
The prior module in Spectral Reconstruction Module
"""


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):

        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class SpatialAttention(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(SpatialAttention, self).__init__()
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

    def forward(self,x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)

        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class SpatialPrior(nn.Module):

    def __init__(self, inplanes, planes, bias=True):
        super(SpatialPrior, self).__init__()
        self.conv1 = nn.Conv2d(hs_channels*2, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(planes, inplanes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(hs_channels+ms_channels, hs_channels, 3, 1, 1)
        self.se = SpatialAttention(inplanes, inplanes)
        self.noise_func = FeatureWiseAffine(16, inplanes, False)
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(16),
            nn.Linear(16, 16 * 4),
            Swish(),
            nn.Linear(16 * 4, 16)
        )

        self.block1 = Block(hs_channels, hs_channels, groups=1)
        self.block2 = Block(hs_channels, hs_channels, groups=1, dropout=0)
        self.block3 = Block(hs_channels, hs_channels, groups=1, dropout=0)
        self.res_conv = nn.Conv2d(hs_channels, hs_channels, 1)

    def forward(self, x, P, H, time_emb, condition): # P: prior of itself H: prior from the other branch
        t = self.noise_level_mlp(time_emb)

        y1 = self.block1(x+P)
        y1 = self.noise_func(y1, t)
        y1 = self.block2(y1)
        x = y1 + self.res_conv(x) + H
        x = self.se(x)
        x1 = self.conv3(torch.cat((condition, x), dim=1))

        y2 = self.block2(x1 + P)
        y2 = self.noise_func(y2, t)
        y2 = self.block2(y2)
        x2 = y2 + self.res_conv(x1) + H
        x2 = self.se(x2)
        x2 = self.conv3(torch.cat((condition, x2), dim=1))

        y3 = self.block3(x2 + P)
        y3 = self.noise_func(y3, t)
        y3 = self.block2(y3)
        x3 = y3 + self.res_conv(x2) + H
        x3 = self.se(x3)
        x3 = self.conv3(torch.cat((condition, x3), dim=1))

        return x3


"""
The data integrity in Spectral Reconstruction Module
"""


class SpatDataIntegrity(nn.Module):
    def __init__(self):
        super(SpatDataIntegrity, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.D = nn.Sequential(
            nn.Conv2d(hs_channels, hs_channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(hs_channels, hs_channels, 4, 4)
        )
        self.DT = nn.Sequential(
            nn.ConvTranspose2d(hs_channels, hs_channels, 4, 4),
            nn.PReLU(),
            nn.ConvTranspose2d(hs_channels, hs_channels, 3, 1, 1)
        )

    def forward(self, H, X, Y):
        IN1 = self.D(H) - Y
        IN2 = (H - X) * self.alpha
        IN = (self.DT(IN1) + IN2) * self.beta
        H_T = H - IN
        return H_T
