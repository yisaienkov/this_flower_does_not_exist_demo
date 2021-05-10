import torch


def get_noise(
    n_samples: int, z_dim: int, device: torch.device
) -> torch.Tensor:

    return torch.randn(n_samples, z_dim, device=device)
