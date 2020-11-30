import numpy as np
import torch
import cv2

import albumentations as albu
from albumentations.pytorch import ToTensor


def eval_transforms(image_size=24):
    return albu.Compose([albu.Resize(image_size, image_size, p=1),
                         albu.Normalize(),
                         ToTensor()])


class Estimator():
    def __init__(
            self,
            loaded_model='traced_cnn.zip',
            clusters='clusters.npy'
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loaded_model = torch.jit.load(loaded_model).to(self.device).eval()
        self.clusters = np.load(clusters)

    def openEyeCheck(self, inpIm):
        image = cv2.imread(inpIm)

        result = {"image": image}
        result = eval_transforms()(**result)
        with torch.no_grad():
            inputs = result['image'].unsqueeze(0).to(self.device)
            emb = self.loaded_model(inputs).cpu().numpy().squeeze()

        opened_similarity = np.dot(emb, self.clusters[1]) / (np.linalg.norm(emb) * np.linalg.norm(self.clusters[1]))
        closed_similarity = np.dot(emb, self.clusters[0]) / (np.linalg.norm(emb) * np.linalg.norm(self.clusters[0]))

        softmax_opened = np.exp(opened_similarity) / (np.exp(opened_similarity) + np.exp(closed_similarity))
        # softmax_closed = np.exp(closed_similarity)/(np.exp(opened_similarity)+np.exp(closed_similarity))
        return softmax_opened
