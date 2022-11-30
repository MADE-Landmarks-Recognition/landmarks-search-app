import torch
import numpy as np
from torchvision import transforms
from .knn import KNN  # noqa


class Recognizer:
    def __init__(
        self, checkpoint_path: str, dump_path: str, device: str, image_size=(224, 224)
    ) -> None:
        self.device = device
        self._model = self._load_model(checkpoint_path)
        self.knn = self._load_dump(dump_path)
        self._embedding_size = self.get_embedding(
            torch.zeros((1, 3, 224, 224)).to(device)
        ).shape[-1]
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, checkpoint_path: str):
        net = torch.jit.load(checkpoint_path, map_location=self.device)
        net.eval()
        return net

    def _load_dump(self, dump_path: str):
        return KNN(dump_path=dump_path)

    def get_embedding(self, input_tensor: torch.Tensor):
        with torch.no_grad():
            embedding = self._model(input_tensor)
        return embedding

    def find_similar(self, img: np.ndarray, k=5):
        input_tensor = self._transform(img).unsqueeze(0).to(self.device)
        embedding = self.get_embedding(input_tensor)
        sims, ids, names = self.knn.search(embedding.cpu().numpy())
        return sims, ids, names
