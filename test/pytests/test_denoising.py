import cv2

from src.denoising import denoising_cv2, metrics

image = cv2.imread("test\\data\\lenna.png")
image_noisy = cv2.imread("test\\data\\lenna_noise.png")
denoiser_cv2 = denoising_cv2.DenoisingCV2()

def test_denoise():
    psnr_before = metrics.psnr(image, image_noisy)
    image_retrieved = denoiser_cv2.denoise(image)
    psnr_after = metrics.psnr(image, image_retrieved)
    assert psnr_before < psnr_after
