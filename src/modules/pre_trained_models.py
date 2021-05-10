from torch.utils import model_zoo
import yaml

from .models import Model


def get_model(model_name: str, device: str = "cpu") -> Model:
    with open('resources/pre_trained_models.yaml') as f:
        models = yaml.safe_load(f)


    model = Model(device=device)
    state_dict = model_zoo.load_url(
        models[model_name]["url"], progress=True, map_location=device
    )

    model.load_state_dict(state_dict)

    return model