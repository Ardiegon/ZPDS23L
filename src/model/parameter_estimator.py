import torch 
import torch.nn as nn
import torchvision.transforms as transforms

class ParameterEstimator(nn.Module):
    def __init__(self, params_n, est_range = (0,1) ,do_transform = False) -> None:
        super().__init__()
        self.params_n = params_n
        self.do_transform = do_transform
        self.est_range = est_range
        self.transform = transforms.Compose([
            transforms.RandomCrop(size=128),
            transforms.ToTensor()
            ]
        )
        
        self.conv1 = nn.Conv2d(3, 32, (5,5), stride=(2,2), padding=(2,2)) # output 64x64x32
        self.conv2 = nn.Conv2d(32, 64, (5,5), stride=(2,2), padding=(2,2)) # output 32x32x64
        self.conv3 = nn.Conv2d(64, 64, (5,5), stride=(2,2), padding=(2,2)) # output 16x16x64
        self.conv4 = nn.Conv2d(64, 64, (5,5), stride=(2,2), padding=(2,2)) # output 8x8x64
        self.conv5 = nn.Conv2d(64, 64, (5,5), stride=(2,2), padding=(2,2)) # output 4x4x64
        self.lin = nn.Linear(1024, self.params_n)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.do_transform:
            x = self.transform(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x.reshape(x.shape[0], -1)
        x = self.sigmoid(self.lin(x)) * self.est_range[1]-self.est_range[0] + self.est_range[0]
        return x
    
class ParameterEstimatorLight(nn.Module):
    def __init__(self, params_n, est_range = (0,1), do_transform = False) -> None:
        super().__init__()
        self.params_n = params_n
        self.do_transform = do_transform
        self.est_range = est_range
        self.transform = lambda x: x[:,:,:64, :64]
        
        self.conv1 = nn.Conv2d(3, 16, (5,5), stride=(4,4), padding=(2,2)) # output 8x8x64
        self.conv2 = nn.Conv2d(16, 64, (5,5), stride=(4,4), padding=(2,2)) # output 4x4x64
        self.drop = nn.Dropout(0.5)
        self.lin = nn.Linear(1024, self.params_n)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def get_reg_loss_l2(self):
        regularization_loss = sum(p.norm(2) for p in self.parameters()) / sum(p.numel() for p in self.parameters())
        return regularization_loss

    def forward(self, x):
        if self.do_transform:
            x = self.transform(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.drop(x)
        x = self.lin(x)
        # x = self.sigmoid(x) * (self.est_range[1]-self.est_range[0]) + self.est_range[0]
        return x