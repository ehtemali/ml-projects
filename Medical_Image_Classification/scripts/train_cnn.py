import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets

# Data transforms
transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
train_data = datasets.ImageFolder('../data/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# Simple CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(nn.Linear(32*64*64,128), nn.ReLU(), nn.Linear(128,2))
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNN()
