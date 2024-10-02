import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 13 * 13, 32)  # 13x13 car les images sont réduites après deux couches de maxpool
        self.fc2 = nn.Linear(32, num_classes)

    def get_model(self):
        return self.model

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 13 * 13)              # Applatir le tenseur
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
