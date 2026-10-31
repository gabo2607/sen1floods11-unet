import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2x2 seguido de DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling, concatenación con la conexión de salto y DoubleConv"""

    def __init__(self, in_ch, skip_ch, out_ch, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + (skip_ch if use_skip else 0), out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if self.use_skip:
            # Ajusta por si el tamaño de entrada no es múltiplo de 2^depth
            dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net para segmentación binaria de agua sobre Sentinel-1 (VV, VH).

    depth y use_skip quedan expuestos para el estudio de ablación.
    Devuelve logits de forma (B, 1, H, W).
    """

    def __init__(self, in_channels=2, base_channels=32, depth=4, use_skip=True):
        super().__init__()
        chs = [base_channels * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(Down(chs[i], chs[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            Up(chs[i + 1], chs[i], chs[i], use_skip) for i in reversed(range(depth))
        )
        self.outc = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        x = self.outc(x)
        if x.shape[2:] != size:  # solo ocurre sin conexiones de salto y tamaños impares
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x
