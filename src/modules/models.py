from typing import Dict

import numpy as np

from .generators import GeneratorWGAN
from .utils import get_noise


class Model:
    def __init__(self, device):
        self.z_dim = 64
        self.generator = GeneratorWGAN().to(device)
        self.device = device

    def load_state_dict(self, state_dict: Dict) -> None:
        self.generator.load_state_dict(state_dict["model_state_dict"])

    def eval(self):
        self.generator.eval()

    def __call__(self) -> np.ndarray:
        self.eval()

        with torch.no_grad():
            fake_noise = get_noise(1, self.z_dim, device=self.device)
            fake = self.generator(fake_noise)[0]

            image_tensor = (fake + 1) / 2
            image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

            return image



    