import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os
import pickle

import pandas as pd
from scipy import spatial
from scipy.spatial import cKDTree
from app_utils import get_model, get_faiss_index, get_mapping
from torchvision import transforms


# config
IMG_DIR = "./data/train10k/"
UPL_DIR = "./data/uploaded/"
IMG_SIZE = 256
CROP_SIZE = 224
TOP_K = 5
DEVICE = "cuda"


st.set_page_config(
    layout="wide",
    page_title='Landmarks',
)


@st.cache(allow_output_mutation=True)
def load_model():
    model = get_model()
    return model


@st.cache(allow_output_mutation=True)
def load_faiss_index():
    faiss_index = get_faiss_index()
    return faiss_index


@st.cache(allow_output_mutation=True)
def load_mapping():
    mapping = get_mapping()
    return mapping


def get_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def transform_image(image):
    trans = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    ])
    return trans(image)



def st_get_top_similar(image_emb, faiss_index, mapping, top_n=10):
    _, prediction = faiss_index.search(image_emb.reshape(1, -1), k=top_n)
    top_similar = {
        "names": [],
        "ids": []
    }

    for idx in prediction[0]:
        top_similar["ids"].append(mapping[idx][0])
        name = mapping[idx][1]
        image_path = f"{IMG_DIR}{name[0]}/{name[1]}/{name[2]}/{name}"
        top_similar["names"].append(image_path)

    return top_similar


def st_get_image_emb(image, emb_model):
    image_emb = emb_model(image.to(DEVICE).unsqueeze(0))
    return image_emb



@st.cache(allow_output_mutation=True)
def get_df_ids():
    current_path = os.getcwd()
    label_to_category = os.path.join(current_path, 'data/train10k/train_label_to_category.csv')
    train10k_ids = os.path.join(current_path, 'data/train10k/train10k.csv')
    df = pd.read_csv(label_to_category)
    df_ids = pd.read_csv(train10k_ids)
    return df, df_ids


def main():
    st.title('Landmarks retrieval')
    st.subheader('MADE-2022')

    ### Load datasets
    df, df_ids = get_df_ids()
    st.write('Datasets loaded:', df.shape, df_ids.shape)

    ### Load artifacts
    emb_model = load_model()
    faiss_index = load_faiss_index()
    mapping = load_mapping()

    ### Load image
    image_file = st.file_uploader('Upload your Image', type=['jpg','png', 'jpeg'])
    if not image_file:
        return None
    image = Image.open(image_file)
    save_path = os.path.join(UPL_DIR, image_file.name)
    image.save(save_path)

    # check that all is ok
    image = get_image(save_path)
    st.image(image, caption='Uploaded Image without CROP and RESIZE.', use_column_width=False)
    image = transform_image(image)
    image_emb = st_get_image_emb(image, emb_model)

    # mapping -> df, df_ids
    top_similar = st_get_top_similar(image_emb, faiss_index, mapping, top_n=TOP_K)
    st.write('Similar images:', top_similar["ids"])
    st.image(top_similar["names"], top_similar["ids"])


if __name__ == '__main__':
    main()

