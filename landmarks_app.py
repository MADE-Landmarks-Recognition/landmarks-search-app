import os

import numpy as np
import streamlit as st
from PIL import Image

from src.model import Recognizer


# config
IMG_DIR = "../data/train10k/"
UPL_DIR = "../data/uploaded/"
IMG_SIZE = 224
TOP_K = 5
DEVICE = "cuda"
CHECKPOINT_PATH = "./checkpoints/extractor.torchscript"
DUMP_PATH = "./checkpoints/landmarks_db"

if not os.path.exists(UPL_DIR):
    os.makedirs(UPL_DIR)

st.set_page_config(
    layout="wide",
    page_title="Landmarks",
)


@st.cache(allow_output_mutation=True)
def load_model():
    return Recognizer(CHECKPOINT_PATH, DUMP_PATH, DEVICE, (IMG_SIZE, IMG_SIZE))


def get_image(path):
    image = Image.open(path).convert("RGB")
    return image


def st_get_top_similar(
    image: np.ndarray,
    recognizer: Recognizer,
    k=TOP_K,
):
    sims, ids, names = recognizer.find_similar(image, k=k)
    names = [f"{IMG_DIR}{name[0]}/{name[1]}/{name[2]}/{name}" for name in names[0]]
    top_similar = {"names": names, "ids": ids[0]}
    return top_similar


def main():
    st.title("Landmarks retrieval")
    st.subheader("MADE-2022")

    ### Load artifacts
    recognizer = load_model()

    ### Load image
    image_file = st.file_uploader("Upload your Image", type=["jpg", "png", "jpeg"])
    if not image_file:
        return None
    image = Image.open(image_file)
    save_path = os.path.join(UPL_DIR, image_file.name)
    image.save(save_path)

    # check that all is ok
    image = get_image(save_path)
    st.image(
        image, caption="Uploaded Image without CROP and RESIZE.", use_column_width=False
    )

    top_similar = st_get_top_similar(image, recognizer, k=TOP_K)
    st.write("Similar images:", top_similar["ids"])
    st.image(top_similar["names"], top_similar["ids"])


if __name__ == "__main__":
    main()
