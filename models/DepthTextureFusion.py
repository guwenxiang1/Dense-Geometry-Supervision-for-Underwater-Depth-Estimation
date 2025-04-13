import torch
from torch import nn
import torch.nn.functional as F


def norm_layer(channel, norm_name='gn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class QualityTCalculation(nn.Module):
    def __init__(self, channel):
        super(QualityTCalculation, self).__init__()

        self.conv_rgb = nn.Conv2d(channel, channel, 3, padding=1)  
        self.conv_depth = nn.Conv2d(channel, channel, 3, padding=1)  
        
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, 1)
        )

        self.softmax = nn.Softmax(dim=1)  

    def forward(self, rgb, depth):
        rgb_feat = self.conv_rgb(rgb)
        depth_feat = self.conv_depth(depth)


        combined_feat = rgb_feat * depth_feat  


        combined_feat = self.softmax(combined_feat)

        pooled_feat = combined_feat.mean(dim=(2, 3))  
        quality_t = self.mlp(pooled_feat)

        return quality_t


class DepthTextureFusion(nn.Module):
    def __init__(self, channel, dilation, kernel=5):
        super(DepthTextureFusion, self).__init__()

        self.spatial_att_1 = SpatialAttention()
        self.spatial_att_2 = SpatialAttention()
        
        self.channel_att_1 = ChannelAttention(channel=channel)
        self.channel_att_2 = ChannelAttention(channel=channel)

        self.depth_refine = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
        )
        self.rgb_refine = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
        )

        self.quality_t_calculation = QualityTCalculation(channel)

    def forward(self, depth, rgb):
        quality_t = self.quality_t_calculation(rgb, depth)
        depth_1 = depth * self.spatial_att_1(rgb)
        rgb_1 = rgb * self.spatial_att_2(depth)

        depth_1 = depth_1 * self.channel_att_1(depth_1)
        rgb_1 = rgb_1 * self.channel_att_2(rgb_1)

        fused = depth_1 + (quality_t * rgb_1)

        depth_ret = depth + self.depth_refine(fused)
        rgb_ret = rgb + self.rgb_refine(fused)

        return depth_ret, rgb_ret, fused


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 4, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out)
        return self.sigmoid(x)

