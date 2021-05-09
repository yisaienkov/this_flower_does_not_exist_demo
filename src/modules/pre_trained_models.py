from torch.utils import model_zoo

from .models import Model


models = {
    "2021-05-09_wgan_generator": {
        "url": "",
        "model": Model,
    }
}


def get_model(model_name: str, device: str = "cpu") -> Model:
    model = models[model_name]["model"](device=device)
    state_dict = model_zoo.load_url(
        models[model_name]["url"], progress=True, map_location=device
    )

    model.load_state_dict(state_dict)

    return model