# Model using denoising from OpenCV library

import cv2
import numpy as np
import torch

from src.denoising.denoising_model_interface import DenoisingModelInterface

class DenoisingCV2(DenoisingModelInterface):

    def denoise(self, img: np.array, lum_param = 10, clr_param = 10) -> np.array:
        dst = cv2.fastNlMeansDenoisingColored(img, None, lum_param , clr_param, 9, 15)
        return dst
    

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)
    parser.add_argument("--p", type=float)
    args = parser.parse_args()
    image_path = args.i
    param = args.p
    dn = DenoisingCV2()
    cv2.imwrite("result.png", dn.denoise(cv2.imread(image_path), lum_param=param, clr_param=param))

if __name__ == "__main__":
    test()