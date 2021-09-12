import random
from typing import Dict, List

import numpy as np
import torch
from omegaconf import DictConfig

from .generators import GeneratorWGAN
from .utils import get_truncated_noise, get_one_hot_labels, combine_vectors


class Model:
    def __init__(self, cfg: DictConfig, device: str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.generator = GeneratorWGAN(
            input_dim=self.cfg.model.z_dim + self.cfg.model.n_classes,
            im_chan=self.cfg.model.image_channels,
        )
        self.generator.to(self.device)

    def load_state_dict(self, state_dict: Dict) -> None:
        self.generator.load_state_dict(state_dict["model_state_dict"])

    def eval(self) -> None:
        self.generator.eval()

    def __call__(self, n_samples: int, classes: List[str], truncation: float) -> np.ndarray:
        self.eval()

        with torch.no_grad():
            fake_noise = get_truncated_noise(
                n_samples=n_samples,
                z_dim=self.cfg.model.z_dim,
                truncation=truncation,
                device=self.device,
            )
            elements = list(
                map(lambda x: self.cfg.classes[x], random.choices(classes, k=n_samples))
            )
            one_hot_labels = get_one_hot_labels(torch.tensor(elements), self.cfg.model.n_classes)
            one_hot_labels.to(self.device)
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)

            fake = self.generator(noise_and_labels)

            image_tensor = (fake + 1) / 2

            image = image_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)

            return image
