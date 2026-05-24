import math

import torch
from torch import nn


class ECAAttention(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x)
        weights = weights.squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights)
        weights = self.sigmoid(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights.expand_as(x)


class GroupBatchNorm2d(nn.Module):
    def __init__(self, channels: int, group_num: int = 16, eps: float = 1e-10) -> None:
        super().__init__()
        group_num = min(group_num, channels)
        while channels % group_num != 0 and group_num > 1:
            group_num -= 1

        self.group_num = group_num
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_grouped = x.view(n, self.group_num, -1)
        mean = x_grouped.mean(dim=2, keepdim=True)
        var = x_grouped.var(dim=2, keepdim=True, unbiased=False)
        x_norm = ((x_grouped - mean) / torch.sqrt(var + self.eps)).view(n, c, h, w)
        return x_norm * self.weight + self.bias


class SRU(nn.Module):
    """Spatial reconstruction unit used in SCConv."""

    def __init__(self, channels: int, group_num: int = 16, gate_threshold: float = 0.5) -> None:
        super().__init__()
        self.gn = GroupBatchNorm2d(channels, group_num=group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gn_x = self.gn(x)
        weight = self.gn.weight / torch.sum(self.gn.weight)
        reweights = self.sigmoid(gn_x * weight)

        info_mask = reweights >= self.gate_threshold
        non_info_mask = reweights < self.gate_threshold
        x_info = info_mask * x
        x_non_info = non_info_mask * x
        return self._reconstruct(x_info, x_non_info)

    @staticmethod
    def _reconstruct(x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        x_11, x_12 = torch.chunk(x_1, 2, dim=1)
        x_21, x_22 = torch.chunk(x_2, 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    """Channel reconstruction unit used in SCConv."""

    def __init__(
        self,
        channels: int,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        upper_channels = int(alpha * channels)
        lower_channels = channels - upper_channels
        upper_squeezed = max(1, upper_channels // squeeze_ratio)
        lower_squeezed = max(1, lower_channels // squeeze_ratio)

        self.up_channel = upper_channels
        self.low_channel = lower_channels
        self.squeeze1 = nn.Conv2d(upper_channels, upper_squeezed, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(lower_channels, lower_squeezed, kernel_size=1, bias=False)
        self.gwc = nn.Conv2d(
            upper_squeezed,
            channels,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=self._valid_groups(upper_squeezed, group_size),
            bias=False,
        )
        self.pwc1 = nn.Conv2d(upper_squeezed, channels, kernel_size=1, bias=False)
        self.pwc2 = nn.Conv2d(lower_squeezed, channels - lower_squeezed, kernel_size=1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _valid_groups(channels: int, group_size: int) -> int:
        groups = min(group_size, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upper, lower = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        upper = self.squeeze1(upper)
        lower = self.squeeze2(lower)

        y1 = self.gwc(upper) + self.pwc1(upper)
        y2 = torch.cat([self.pwc2(lower), lower], dim=1)
        out = torch.cat([y1, y2], dim=1)

        weights = torch.softmax(self.global_avg_pool(out), dim=1)
        out = out * weights
        out1, out2 = torch.chunk(out, 2, dim=1)
        return out1 + out2


class SCConv(nn.Module):
    """Spatial and Channel Reconstruction Convolution."""

    def __init__(
        self,
        channels: int,
        group_num: int = 16,
        gate_threshold: float = 0.5,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.sru = SRU(channels, group_num=group_num, gate_threshold=gate_threshold)
        self.cru = CRU(
            channels,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sru(x)
        x = self.cru(x)
        return x
