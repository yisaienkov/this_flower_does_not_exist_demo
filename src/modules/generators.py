from torch import nn


class GeneratorWGAN(nn.Module):
    def __init__(self, *, input_dim: int, im_chan: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim

        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4, kernel_size=4, padding=3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            self.make_gen_block(hidden_dim, hidden_dim),
            self.make_gen_block(hidden_dim, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, final_layer=True),
        )

    def make_gen_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(len(x), self.input_dim, 1, 1)
        return self.gen(x)
