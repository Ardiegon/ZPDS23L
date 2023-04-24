import cv2
import argparse
import numpy as np

def add_gausian_noise(image: np.array, mean: float, var: float) -> np.array:
    row,col,ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)*255
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to saving location")
    parser.add_argument("--mean", "-m", type=float, default=0.0, help="Mean of Gaussian distribution")
    parser.add_argument("--var", "-v", type=float, default=1.0, help="Variation of Gaussian distribution")
    return parser.parse_args()

def main(args):
    image = cv2.imread(args.input)
    image = add_gausian_noise(image, args.mean, args.var)
    cv2.imwrite(args.output, image)

if __name__ == "__main__":
    args = parse_args()
    main(args)

