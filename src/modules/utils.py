import torch


def get_noise(
    n_samples: int, z_dim: int, device: str = "cpu"
) -> torch.Tensor:

    return torch.randn(n_samples, z_dim, device=device)
