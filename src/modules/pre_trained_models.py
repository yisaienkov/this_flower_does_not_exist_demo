from torch.utils import model_zoo
from omegaconf import DictConfig

from .models import Model


def get_model(cfg: DictConfig, device: str = "cpu") -> Model:
    model = Model(cfg=cfg, device=device)
    state_dict = model_zoo.load_url(cfg.model.url, progress=True, map_location=device)
    model.load_state_dict(state_dict)

    return model