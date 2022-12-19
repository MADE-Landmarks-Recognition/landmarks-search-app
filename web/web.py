import os
import requests
import pandas as pd
import streamlit as st
from PIL import Image
# import re


# config
IMG_DIR = "../data/filtered/"                    # common data for all services
UPL_DIR = "./data/uploaded/"                     # upload dir just for web part
TOP_K = 5                                        # api service provide top 5 now
API_ENDPOINT = os.getenv("API_ENDPOINT_LOCAL")   # API_ENDPOINT_LOCAL local api for testing
MAX_SIZE = (256, 256)


if not os.path.exists(UPL_DIR):
    os.makedirs(UPL_DIR)

st.set_page_config(
    layout="centered",
    page_title="Landmarks"
)

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)


def get_image(path, resize=True):
    image = Image.open(path).convert("RGB")
    if resize:
        image = image.resize(MAX_SIZE, Image.Resampling.BICUBIC)
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

    ### Load image
    with st.sidebar:
        st.title("Landmarks retrieval")
        st.subheader("MADE-2022")
        image_file = st.file_uploader("Upload your Image", type=["jpg", "png", "jpeg"])
        if not image_file:
            return None
        image = Image.open(image_file)
        save_path = os.path.join(UPL_DIR, image_file.name)
        image.save(save_path)
        st.write(f"Image uploaded: : {image_file.name}")
        TOP_K = st.selectbox('Number of images to retrieve: ', (1, 3, 5))
        grid_option = st.selectbox('Set grid width', (False, True))


    with st.container():
        col1, col2 = st.columns(2, gap="small")
    
        # check that all is ok
        image = get_image(save_path)

    # Output original    
    with col1:
        st.subheader("Original image: ")
        st.image(
            image, caption=f"Uploaded Image: {image_file.name}", 
            use_column_width=grid_option,)
    
    # api service part
    top_similar = get_top_similar(image, TOP_K)

    # print table to sidebar
    with st.sidebar:
        st.write("Top similar:")
        df = pd.DataFrame(top_similar, columns=('ids', 'names'))
        st.dataframe(df, use_container_width=True)
    
    # print location
    # uncomment and make sure the csv contains scrapped info for the full DS
    no_print = False
    if no_print:
        try:
            df1 = pd.read_csv('../data/extracted10000_3col.csv')
            id = df.ids[0]
            land_location = df1[df1.id == id]['Location'].to_string()
            l_new = ' '.join(land_location.split()[1:])
            st.text(l_new)

            ### extract country
            
            # country = re.findall('.[^A-Z]*', l_new)[-1]
            # alternatively without re
            # country = "".join([(" "+i if i.isupper() else i) for i in l_new]).strip().split()[-1] 
            # st.text(country)
        except:
            pass

    # top retrieval 
    with col2:
        st.subheader("Top retrieval: ")
        im = get_image(top_similar['paths'][0])
        cap = df['names'][0].strip().replace("_", " ")
        st.image(im, caption=cap, use_column_width=grid_option, width = MAX_SIZE[1])
  
    # retrieved images output
    st.subheader("Retrieved images: ")

    c1, c2 = st.columns([1,1], gap="small")
    for i in range(1, TOP_K):
        im = get_image(top_similar['paths'][i])
        cap = df['names'][i].strip().replace("_", " ")
        if i % 2 != 0:
            
            with c1:
                st.image(im, caption=[cap], use_column_width=grid_option, width = MAX_SIZE[1])
        else:
            with c2:
                st.image(im, caption=[cap], use_column_width=grid_option, width = MAX_SIZE[1])


if __name__ == "__main__":
    main()
