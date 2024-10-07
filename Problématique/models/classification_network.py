import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=72, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=72, out_channels=72, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(72)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(72 * 6 * 6, 48)
        self.batchnorm3 = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, num_classes)

    def get_model(self):
        return self.model

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.batchnorm1(x)
        x = self.pool3(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        x = self.batchnorm2(x)
        x = x.view(-1, 72 * 6 * 6)              # Applatir le tenseur
        x = torch.relu(self.fc1(x))
        x = self.batchnorm3(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
