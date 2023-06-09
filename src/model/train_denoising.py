import os
import torch
import torch.nn as nn
import logging

from tqdm import tqdm
from model.dataset import ImageDataset
from model.parameter_estimator import ParameterEstimator, ParameterEstimatorLight
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def setup_logger(log_file_path):
    # Create a logger and set the logging level
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the formatting
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def init_opts():
    div2k_path = r"C:\Users\oskbs\Documents\Datasets\DIV2K\train"
    device = "cuda"
    dataset = ImageDataset(div2k_path, crop_size=64)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
    loss = nn.MSELoss()
    model = ParameterEstimatorLight(params_n=1, est_range=(0,0.05)).to(device)
    return {
        "device": device,
        "model": model,
        "loss": loss,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }

def train(opts):
    logger = setup_logger(os.path.join("src", "checkpoints", "train_denoise_log.txt"))
    model = opts["model"]
    criterion = opts["loss"]
    device = opts["device"]
    train_loader = opts["train_loader"]
    test_loader = opts["test_loader"]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    n_epoch = 40001

    for epoch in  tqdm(range(n_epoch), desc="Epochs", position=0):
        running_train_loss = 0.0
        running_test_loss = 0.0

        model.train()
        
        for noisy, std in train_loader:
            std = std.to(device)
            noisy = noisy.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            
            loss = criterion(std, outputs) + 0.1 * model.get_reg_loss_l2()
            loss.backward()
            
            optimizer.step()
            running_train_loss += loss.item() * noisy.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        logger.debug(f"Epoch [{epoch+1}/{n_epoch}]")
        logger.debug(f"Train Loss: {epoch_train_loss}")

        logger.debug(std[:5].tolist())
        logger.debug(outputs[:5].tolist())

        model.eval()

        for noisy, std in test_loader:
            std = std.to(device)
            noisy = noisy.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            
            loss = criterion(std, outputs)
            running_test_loss += loss.item() * noisy.size(0)
        
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        logger.debug(f"Test Loss: {epoch_test_loss}")
        logger.debug(std[:5].tolist())
        logger.debug(outputs[:5].tolist())

        if epoch%1000==0:
            torch.save(model.state_dict(), os.path.join("src", "checkpoints", f"model_denoise_{epoch}.pth"))


def main():
    opts = init_opts()
    train(opts)

if __name__ == "__main__":
    main()