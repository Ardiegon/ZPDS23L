# Model using denoising from OpenCV library and torch model for parameter estimation

import os
import cv2
import numpy as np
import torch

from src.denoising.denoising_model_interface import DenoisingModelInterface
from model.parameter_estimator import ParameterEstimatorLight
from model.mapping_params import get_std_to_noise_param_map

class DenoisingHybrid(DenoisingModelInterface):
    def __init__(self, device = "cuda", bias = 0.0) -> None:
        super().__init__()
        self.device = device
        self.post_estimation_bias = bias
        param_est_model = ParameterEstimatorLight(est_range=(0,0.05), params_n=1, do_transform=True)
        param_est_model.load_state_dict(torch.load(os.path.join("src", "checkpoints", "model_denoise_40000.pth")))
        self.param_est_model = param_est_model.to(device)
        self.map_model_to_param_func = get_std_to_noise_param_map()


    def denoise(self, img: np.array) -> np.array:
        sample = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(torch.float32).to(self.device)
        output = self.param_est_model(sample).detach().cpu() + self.post_estimation_bias
        print(output.item())
        param = self.map_model_to_param_func(output.item())
        print(param)
        dst = cv2.fastNlMeansDenoisingColored(img, None, param, param, 9, 15)
        return dst
    

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)
    args = parser.parse_args()
    image_path = args.i
    dn = DenoisingHybrid()
    cv2.imwrite("result.png", dn.denoise(cv2.imread(image_path)))

if __name__ == "__main__":
    test()