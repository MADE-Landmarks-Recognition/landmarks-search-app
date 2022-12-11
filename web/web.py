import os
import requests

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# config
IMG_DIR = "../data/train10k/"                    # common data for all services
UPL_DIR = "./data/uploaded/"                     # upload dir just for web part
TOP_K = 5                                        # api service provide top 5 now
API_ENDPOINT = os.getenv("API_ENDPOINT_LOCAL")   # local api for testing


if not os.path.exists(UPL_DIR):
    os.makedirs(UPL_DIR)

st.set_page_config(
    layout="centered",
    page_title="Landmarks"
)


def get_image(path):
    image = Image.open(path).convert("RGB")
    return image


def get_top_similar(image, k):
    """
        send request to landmarks-api service
            {
                "top_k": top k images (now provided top 5)
                "size": image size (need for recovering image from bytes)
                "image": image in bytes
            }
        get top similar json
            {
                "ids": [],
                "names": [],
                "paths": []
            }
    """
    data = {
        'top_k': k,
        'size': image.size,
        'image': image.tobytes().decode("latin-1")
    }
    response = requests.post(url=API_ENDPOINT, json=data, timeout=20)
    top_similar = response.json()
    return top_similar


def main():
    st.title("Landmarks retrieval")
    st.subheader("MADE-2022")

    ### Load image
    image_file = st.file_uploader("Upload your Image", type=["jpg", "png", "jpeg"])
    if not image_file:
        return None
    image = Image.open(image_file)
    save_path = os.path.join(UPL_DIR, image_file.name)
    image.save(save_path)

    col1, col2 = st.columns(2)

    # check that all is ok
    image = get_image(save_path)
    with col1:
        st.image(
            image, caption=f"Uploaded Image: {image_file.name}",
            use_column_width=True,
        )

    # api service part
    top_similar = get_top_similar(image, k=TOP_K)

    with col2:
        st.write("Top similar:")
        df = pd.DataFrame(top_similar, columns=('ids', 'names'))
        st.dataframe(df, use_container_width=False)

    # st.image(top_similar["paths"], top_similar["ids"], width=224)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(top_similar["paths"][0])
        st.write(top_similar["ids"][0])

    with col2:
        st.image(top_similar["paths"][1])
        st.write(top_similar["ids"][1])

    with col3:
        st.image(top_similar["paths"][2])
        st.write(top_similar["ids"][2])
    
    with col4:
        st.image(top_similar["paths"][3])
        st.write(top_similar["ids"][3])

    with col5:
        st.image(top_similar["paths"][4])
        st.write(top_similar["ids"][4])


if __name__ == "__main__":
    main()
