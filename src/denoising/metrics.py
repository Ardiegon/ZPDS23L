# script for measuring supervised noise in two images.

import cv2
import argparse
import numpy as np

from enum import Enum
from math import log10, sqrt

class Metrics(Enum):
    PSNR = "psnr"
    MSE = "mse"

def get_metric(metric: Metrics) -> function:
    metric_dict = {
        Metrics.PSNR: psnr,
        Metrics.MSE: mse
    }
    return metric_dict[metric]

def mse(gt, sample):
    mse = np.mean((gt - sample) ** 2)
    return mse

def psnr(gt, sample):
    """
    Peak Signal-Noise Ratio:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = mse(gt, sample)
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", type=Metrics, default="psnr", help="Path to ground truth image")
    parser.add_argument("--gt", "-i", type=str, required=True, help="Path to ground truth image")
    parser.add_argument("--sample", "-o", type=str, required=True, help="Path to sample image")
    return parser.parse_args()

def main(args):
    gt = cv2.imread(args.gt)
    sample = cv2.imread(args.saple)
    metric = get_metric(args.metric)
    metric_result = metric(gt, sample)
    print(metric_result)

if __name__ == "__main__":
    args = parse_args()
    main(args)