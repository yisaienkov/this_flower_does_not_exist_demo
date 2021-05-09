from torch.utils import model_zoo

from .models import Model


models = {
    "2021-05-09_wgan_generator": "https://github.com/yisaienkov/this_kulbaba_does_not_exist_demo/releases/download/v0.0.1/2021-05-09_wgan_generator.pt",
}


def get_model(model_name: str, device: str = "cpu") -> Model:
    model = Model(device=device)
    state_dict = model_zoo.load_url(
        models[model_name], progress=True, map_location=device
    )

    model.load_state_dict(state_dict)

    return model