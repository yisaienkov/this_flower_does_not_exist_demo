import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from modules import get_model

icon = Image.open("resources/assets/icon.png")
st.set_page_config(
    page_title="This Kulbaba Does Not Exist", 
    page_icon=icon, 
)

@st.cache()
def cached_model():
    model = get_model("2021-05-09_wgan_generator")
    return model


if __name__ == "__main__":
    st.markdown(
        "<h1 style='text-align: center; color: white;'>THIS KULBABA DOES NOT EXIST</h1>", 
        unsafe_allow_html=True,
    )

    model = cached_model()
    image = model()

    fig = plt.figure(figsize=(5, 5))
    fig.patch.set_facecolor("black")
    plt.imshow(image)
    plt.axis("off")

    st.pyplot(fig)
    st.markdown(
        "<p style='text-align: center; color: white;'><i>Press 'R' to get the new kulbaba</i></p>", 
        unsafe_allow_html=True,
    )

    st.subheader("Project info:")
    st.markdown("Model: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)")
    st.markdown("Image: 28*28 pixels")

    st.subheader("Contact me:")
    st.markdown("Yaroslav Isaienkov <oiuygl@gmail.com>")
    st.markdown(
        "LinkedIn [yisaienkov](https://www.linkedin.com/in/yisaienkov/)"
    )
    st.markdown(
        (
            "Source demo code [this_kulbaba_does_not_exist_demo]"
            "(https://github.com/yisaienkov/this_kulbaba_does_not_exist_demo)"
        )
    )