import os
import glob
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from denoising.noise_simulator import add_gausian_noise
import torch

class ImageDataset(Dataset):
    def __init__(self, folder_path, crop_size=64):
        self.folder_path = folder_path
        self.image_paths = self.load_image_paths()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(size=(crop_size, crop_size))    
            ]
        )

    def add_noise(self, img):
        std = (torch.rand(1)*0.05).to(torch.float32)
        noisy = torch.from_numpy(add_gausian_noise(img.numpy(), mean=0.0, var=std.item())).to(torch.float32)
        return noisy, std

    def load_image_paths(self):
        image_paths = glob.glob(os.path.join(self.folder_path, '*.jpg'))
        image_paths.extend(glob.glob(os.path.join(self.folder_path, '*.png')))
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = self.transform(image)
        noisy, std = self.add_noise(image)
        return noisy, std
