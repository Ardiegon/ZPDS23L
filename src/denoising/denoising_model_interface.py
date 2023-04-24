# Standard interface for other denoising models. Treat it as virtual class.
import numpy as np

class DenoisingModelInterface:
    def denoise(self, img: np.array) -> np.array:
        raise NotImplementedError("Virtual method can not be called.")