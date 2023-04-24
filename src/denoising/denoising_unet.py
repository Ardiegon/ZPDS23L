# PyTorch UNet model for denoising

import cv2
import torch
import torch.nn as nn
import numpy as np

from src.denoising.denoising_model_interface import DenoisingModelInterface

DEVICE = "cuda"

class UnetModel(nn.Module):
    #TODO
    def __init__(self):
        pass

    def forward(self, x):
        pass

class DenoisingUNet(DenoisingModelInterface):
    def __init__(self) -> None:
        self.model = UnetModel().to(DEVICE)

    def image_to_tensor(self, img: np.array) -> torch.Tensor:
        return torch.from_numpy(img/255).to(DEVICE)
    
    def tensor_to_image(self, tensor: torch.Tensor) -> np.array:
        return tensor.cpu().detach().numpy()*255

    def denoise(self, img: np.array) -> np.array:
        tensor = self.image_to_tensor(img)
        tensor = self.model(tensor)
        dst = self.tensor_to_image(tensor)
        return dst
        

if __name__ == "__main__":
    print(torch.cuda.is_available())