import os
from typing import Optional, List

import faiss
import numpy as np


class KNN:
    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        img_names: Optional[List[str]] = None,
        dump_path: Optional[str] = None,
        method="cosine",
    ):
        if not dump_path:
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            self.N = len(embeddings)
            self.D = embeddings[0].shape[-1]
            self.embeddings = (
                embeddings
                if embeddings.flags["C_CONTIGUOUS"]
                else np.ascontiguousarray(embeddings)
            )
            self.labels = labels
            self.img_names = img_names
            self.index = {"cosine": faiss.IndexFlatIP, "euclidean": faiss.IndexFlatL2,}[
                method
            ](self.D)
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                self.index = faiss.index_cpu_to_all_gpus(self.index)
            self.add()
        else:
            dump = np.load(dump_path + ".npz")
            self.index = faiss.read_index(dump_path + ".index")
            self.labels = dump["labels"]
            self.img_names = dump["img_names"]

    def add(self, batch_size=10000):
        """Add data into index"""
        if self.N <= batch_size:
            self.index.add(self.embeddings)
        else:
            [
                self.index.add(self.embeddings[i : i + batch_size])
                for i in range(0, len(self.embeddings), batch_size)
            ]

    def search(self, queries, k=5):
        """Search
        Args:
            queries: query vectors
            k: get top-k results
        Returns:
            sims: similarities of k-NN
            ids: indexes of k-NN
        """
        if not queries.flags["C_CONTIGUOUS"]:
            queries = np.ascontiguousarray(queries)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        sims, ids = self.index.search(queries, k)
        names = [[self.img_names[idx] for idx in id_] for id_ in ids]
        ids = self.labels[ids] if self.labels is not None else ids
        return sims, ids, names

    def dump(self, save_path: str = "landmarks_db"):
        faiss.write_index(self.index, save_path + ".index")
        np.savez_compressed(
            save_path + ".npz", labels=self.labels, img_names=self.img_names
        )
