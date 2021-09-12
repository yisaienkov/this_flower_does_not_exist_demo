import torch
from torch.nn import functional as torch_f
from scipy.stats import truncnorm


def get_truncated_noise(n_samples: int, z_dim: int, truncation: float, device: torch.device):
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise).to(device)


def get_one_hot_labels(labels, n_classes):
    return torch_f.one_hot(labels, n_classes)


def combine_vectors(x, y):
    combined = torch.cat([x.float(), y.float()], dim=1)
    return combined
