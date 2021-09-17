import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf, DictConfig

from modules import get_model

icon = Image.open("resources/assets/icon.png")
st.set_page_config(
    page_title="This Flower Does Not Exist",
    page_icon=icon,
)


@st.cache()
def load_cfg():
    return OmegaConf.load("params.yaml")


@st.cache()
def cached_model(cfg: DictConfig):
    model = get_model(cfg)
    return model


if __name__ == "__main__":

    st.markdown(
        "<h1 style='text-align: center; color: white;'>THIS FLOWER DOES NOT EXIST</h1>",
        unsafe_allow_html=True,
    )

    cfg = load_cfg()
    model = cached_model(cfg)

    flower_type = st.sidebar.radio(
        "What do you want to generate?", ("ALL",) + tuple(cfg.classes.keys())
    )
    if flower_type == "ALL":
        classes = list(cfg.classes.keys())
    else:
        classes = [flower_type]

    truncation = st.sidebar.slider("Fidelity/Diversity trade off (truncation)", 0.01, 3.0, 1.5)
    grid_size = st.sidebar.slider("The grid size", 1, 7, 4)

    images = model(n_samples=grid_size ** 2, truncation=truncation, classes=classes)
    with st.container():
        ind = 0
        for row in range(grid_size):
            for col in st.columns(grid_size):
                with col:
                    st.image(
                        cv2.resize(
                            images[ind], (cfg.model.image_size * 8, cfg.model.image_size * 8), 
                            interpolation=cv2.INTER_NEAREST
                        ),
                        use_column_width="always",
                    )
                    ind += 1

    st.markdown(
        "<p style='text-align: center; color: white;'><i>Press 'R' to get the new flower(s)</i></p>",
        unsafe_allow_html=True,
    )

    st.subheader("The model is based on:")
    st.markdown("[Wasserstein GAN](https://arxiv.org/abs/1701.07875)")
    st.markdown("[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)")
    st.markdown(f"Image size: {cfg.model.image_size} x {cfg.model.image_size} pixels")

    st.subheader("Contact me:")
    st.markdown("Yaroslav Isaienkov <oiuygl@gmail.com>")
    st.markdown("LinkedIn [yisaienkov](https://www.linkedin.com/in/yisaienkov/)")
    st.markdown(
        (
            "Source demo code [this_flower_does_not_exist_demo]"
            "(https://github.com/yisaienkov/this_flower_does_not_exist_demo)"
        )
    )
