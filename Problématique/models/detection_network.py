import torch
import torch.nn as nn
import torch.nn.functional as F
from triton.language import tensor


class DetectionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        self.num_classes = num_classes

        # AlexNet architecture pour localisation

        #1x53x53
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=11, stride=2, padding=2)#24x24x24
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 16x6x6
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=2)#48x12x12
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#128x3x3
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)#96x6x6
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)#96x6x6
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1)#48x6x6
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#48x3x3
        self.fc1 = nn.Linear(48 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 64)
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 3 * 7)

    def get_model(self):
        return self.model

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(self.relu5(self.conv5(x)))
        x = x.view(-1, 48 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.bn1(x)
        x = self.fc3(x)
        N = torch.softmax(x[:, :9], dim=1) # Prédiction de la classe
        C = torch.sigmoid(x[:, 9:]) # Prédiction des coordonnées et de la taille de la boîte

        classes1 = N[:,:3]
        classes2 = N[:,3:6]
        classes3 = N[:,6:]
        x1 = C[:,:4]
        x2 = C[:, 4:8]
        x3 = C[:, 8:12]

        o1 = torch.cat([x1, classes1], dim=1)
        o2 = torch.cat([x2, classes2], dim=1)
        o3 = torch.cat([x3, classes3], dim=1)

        return torch.stack([o1, o2, o3], dim=1)