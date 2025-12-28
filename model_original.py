import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.max_period, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, padding=0) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        time_emb_proj = F.silu(self.time_emb_proj(time_emb))
        h = h + time_emb_proj.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        h = h + self.shortcut(x)
        return h

class DownSample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class Attention(nn.Module):
    def __init__(self, ch, groups=32):
        super().__init__()
        self.ch = ch
        self.groups = groups
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1, padding=0)
        self.proj = nn.Conv2d(ch, ch, 1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, c, h * w).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(c, device=x.device)))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        out = self.proj(out)
        return x + out

class UNet(nn.Module):
    def __init__(self, in_channels, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # time embedding
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.time_emb_linear1 = nn.Linear(time_emb_dim, time_emb_dim * 4)
        self.time_emb_linear2 = nn.Linear(time_emb_dim * 4, time_emb_dim * 4)

        # 输入层
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

        # 下采样64 128 256 512
        self.down1 = ResidualBlock(64, 64, time_emb_dim * 4)
        self.down1_attn = Attention(64)
        self.down_sample1 = DownSample(64)

        self.down2 = ResidualBlock(64, 128, time_emb_dim * 4)
        self.down2_attn = Attention(128)
        self.down_sample2 = DownSample(128)

        self.down3 = ResidualBlock(128, 256, time_emb_dim * 4)
        self.down3_attn = Attention(256)
        self.down_sample3 = DownSample(256)

        self.down4 = ResidualBlock(256, 512, time_emb_dim * 4)
        self.down4_attn = Attention(512)

        # 中间层
        self.mid1 = ResidualBlock(512, 512, time_emb_dim * 4)
        self.mid_attn = Attention(512)
        self.mid2 = ResidualBlock(512, 512, time_emb_dim * 4)

        # 上采样
        self.up1 = ResidualBlock(512 + 256, 256, time_emb_dim * 4)
        self.up1_attn = Attention(256)
        self.up_sample1 = UpSample(512)

        self.up2 = ResidualBlock(256 + 128, 128, time_emb_dim * 4)
        self.up2_attn = Attention(128)
        self.up_sample2 = UpSample(256)

        self.up3 = ResidualBlock(128 + 64, 64, time_emb_dim * 4)
        self.up3_attn = Attention(64)
        self.up_sample3 = UpSample(128)

        self.up4 = ResidualBlock(64 + 64, 64, time_emb_dim * 4)
        self.up4_attn = Attention(64)

        # 输出层
        self.final_norm = nn.GroupNorm(32, 64)
        self.final_conv = nn.Conv2d(64, in_channels, 3, padding=1)

    def forward(self, x, t):
        # time embedding
        time_emb = self.time_emb(t)
        time_emb = F.silu(self.time_emb_linear1(time_emb))
        time_emb = self.time_emb_linear2(time_emb)

        # 输入层
        x0 = self.init_conv(x)
        hs = []

        # 下采样
        x1 = self.down1(x0, time_emb)
        x1 = self.down1_attn(x1)
        x1_down = self.down_sample1(x1)

        x2 = self.down2(x1_down, time_emb)
        x2 = self.down2_attn(x2)
        x2_down = self.down_sample2(x2)

        x3 = self.down3(x2_down, time_emb)
        x3 = self.down3_attn(x3)
        x3_down = self.down_sample3(x3)

        x4 = self.down4(x3_down, time_emb)
        x4 = self.down4_attn(x4)

        hs.append(x3)  # 8x8
        hs.append(x2)  # 16x16
        hs.append(x1)  # 32x32
        hs.append(x0)  # 32x32

        # 中间层
        x_mid = self.mid1(x4, time_emb)
        x_mid = self.mid_attn(x_mid)
        x_mid = self.mid2(x_mid, time_emb)
        h = x_mid

        # 上采样
        # 上采样1：4x4→8x8，x3(8x8)
        h = self.up_sample1(h)
        h = torch.cat([h, hs[0]], dim=1)
        h = self.up1(h, time_emb)
        h = self.up1_attn(h)

        # 上采样2：8x8→16x16，x2(16x16)
        h = self.up_sample2(h)
        h = torch.cat([h, hs[1]], dim=1)
        h = self.up2(h, time_emb)
        h = self.up2_attn(h)

        # 上采样3：16x16→32x32，x1(32x32)
        h = self.up_sample3(h)
        h = torch.cat([h, hs[2]], dim=1)
        h = self.up3(h, time_emb)
        h = self.up3_attn(h)

        # 上采样4：32x32，x0(32x32)
        h = torch.cat([h, hs[3]], dim=1)
        h = self.up4(h, time_emb)
        h = self.up4_attn(h)

        # 输出层
        out = F.silu(self.final_norm(h))
        out = self.final_conv(out)

        return out

def get_model(dataset_name):
    in_channels = config["img_channels"][dataset_name]
    model = UNet(in_channels).to(config["device"])
    return model