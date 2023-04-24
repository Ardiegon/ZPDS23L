# Model using denoising from OpenCV library

import cv2
import numpy as np

from denoising_model_interface import DenoisingModelInterface

class DenoisingCV2:
    def __init__(self) -> None:
        pass

    def denoise(img: np.array) -> np.array:
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        return dst
    


