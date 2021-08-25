import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


"""
    Pre layernorm
"""


class TokenMixing(nn.Module):
    def __init__(self, dim, patch_num, token_mix):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)

        self.tokenmix = MLP(patch_num, token_mix)

    def forward(self, x):
        x = self.layernorm(x)  # bs, n, c -> bs, c ,n
        x = x.transpose(1, 2)
        x = self.tokenmix(x)
        x = x.transpose(1, 2)  # bs,n , c
        return x


class ChannelMixing(nn.Module):
    def __init__(self, dim, channel_mix):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.channelmix = MLP(dim, channel_mix)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.channelmix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, token_mix, channel_mix, img_size=32, patch_size=4, depth=8, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patch = (img_size // patch_size) ** 2

        self.dim = dim
        self.conv = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        self.module_list = nn.ModuleList([])

        for _ in range(depth):
            self.module_list.append(TokenMixing(dim, self.num_patch, token_mix))
            self.module_list.append(ChannelMixing(dim, channel_mix))

        self.layernorm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        bs = x.size(0)
        x = self.conv(x)  # bs, c, h, w

        x = x.view(bs, self.dim, -1)
        x = x.transpose(1, 2)  # bs, (h, w), c

        for mixer_block in self.module_list:
            x = mixer_block(x)

        x = self.layernorm(x)
        x = x.mean(dim=1)

        return self.mlp_head(x)