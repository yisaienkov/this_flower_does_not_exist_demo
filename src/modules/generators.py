import torch
from torch import nn


class GeneratorWGAN(nn.Module):
    def __init__(
        self, z_dim: int = 64, im_chan: int = 3, hidden_dim: int = 64
    ):
        super().__init__()
        self.z_dim = z_dim
        
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(
        self, 
        input_channels: int, 
        output_channels: int, 
        kernel_size: int = 3, 
        stride: int = 2, 
        final_layer: bool = False,
    ) -> nn.Sequential:
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)