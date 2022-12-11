import io

import numpy as np
from PIL import Image
from flask import Flask, request
from flask import make_response

from src.model import Recognizer


# config
IMG_DIR = "../data/train10k/" # common data for all services
TOP_K = 5                     # api service provide top 5 now
IMG_SIZE = 224
DEVICE = "cuda"
CHECKPOINT_PATH = "./checkpoints/extractor.torchscript"
DUMP_PATH = "./checkpoints/landmarks_db"
MAPPING_PATH = "./checkpoints/id_to_name.mapping"


app = Flask(__name__)

recognizer = Recognizer(CHECKPOINT_PATH, DUMP_PATH, MAPPING_PATH, DEVICE, (IMG_SIZE, IMG_SIZE))


def get_top_similar(
    image: np.ndarray,
    recognizer: Recognizer,
    k=TOP_K,
):
    sims, ids, paths, names = recognizer.find_similar(image, k=k)
    paths = [f"{IMG_DIR}{path[0]}/{path[1]}/{path[2]}/{path}" for path in paths[0]]
    top_similar = {"paths": paths, "ids": ids[0].tolist(), "names": names}
    return top_similar


@app.route('/api/top_k', methods=['POST'])
def top_k():
    data = request.json
    image = Image.frombytes('RGB', data['size'], data['image'].encode('latin-1'), 'raw')
    top_similar = get_top_similar(image, recognizer, data['top_k'])
    return make_response(top_similar, 200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
