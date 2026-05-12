
class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect"
        )

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect"
        )

        self.relu = nn.LeakyReLU(0.1, inplace=True)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, skip_scale=0.7):
        super().__init__()

        self.skip_scale = skip_scale

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2
        )

        self.conv = ResidualConvBlock(
            in_ch=out_ch * 2,
            out_ch=out_ch,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x = torch.cat([self.skip_scale * x2, x1], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_features=32):
        super().__init__()

        f = base_features

        self.inc = ResidualConvBlock(
            in_channels,
            f,
            kernel_size=5,
            padding=2
        )

        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResidualConvBlock(f, f * 2)
        )

        self.down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResidualConvBlock(f * 2, f * 4)
        )

        self.down3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResidualConvBlock(f * 4, f * 8)
        )

        self.up1 = UpBlock(f * 8, f * 4, skip_scale=0.7)
        self.up2 = UpBlock(f * 4, f * 2, skip_scale=0.7)
        self.up3 = UpBlock(f * 2, f, skip_scale=0.7)

        self.outc = nn.Conv2d(
            f,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )

        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        decoded_input = x

        pad = 16
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        residual = self.outc(x)
        residual = residual[..., pad:-pad, pad:-pad]

        enhanced = decoded_input + self.res_scale * residual

        if self.training:
            return enhanced

        return torch.clamp(enhanced, 0.0, 1.0)
