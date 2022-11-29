from pathlib import Path
from pprint import pprint

import torch 
import yaml
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator, precision_at_k
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
import faiss

from retrieval.engine.evaluate import evaluate, get_tester
from retrieval.models.net import RetrievalNet
from retrieval.getter import Getter


DEVICE = "cuda"
EMB_SIZE = 384
CKPT_PATH = Path("./checkpoints/deit_roadmap.ckpt")
FAISS_PATH = Path("./checkpoints/index.faiss")
getter = Getter()


def get_model():
    net = RetrievalNet(
        "vit_deit_distilled",
        embed_dim=384,
        norm_features=False,
        without_fc=True,
        with_autocast=False,
    )
    weights = torch.load(CKPT_PATH, map_location=DEVICE)["net_state"]
    net.load_state_dict(weights)
    net.to(DEVICE)
    net.eval()
    return net


def get_faiss_index():
    # dump example:
    # index = faiss.IndexFlatL2(512)
    # index2 = faiss.IndexIDMap(index)
    # index2.add_with_ids(train_embeddings.numpy(), train_targets.cpu().numpy())
    # faiss.write_index(index, FAISS_PATH)

    # load index:
    faiss_index = faiss.IndexFlatL2(EMB_SIZE)
    faiss_index = faiss.read_index(FAISS_PATH)
    return faiss_index


def get_mapping():
    pass
