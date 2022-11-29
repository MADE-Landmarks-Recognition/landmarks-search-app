from pathlib import Path
from pprint import pprint

import torch 
import yaml
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator, precision_at_k
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from retrieval.engine.evaluate import evaluate, get_tester
from retrieval.models.net import RetrievalNet
from retrieval.getter import Getter


DEVICE = "cuda"
CKPT_PATH = Path("./checkpoints/deit_roadmap.ckpt")
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
    pass


def get_mapping():
    pass