import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        self.num_classes = num_classes

        # AlexNet architecture pour localisation

        #1x53x53
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11, stride=2, padding=2)#32x24x24
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32x6x6
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2)#128x12x12
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#128x3x3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)#256x6x6
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)#256x6x6
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)#128x6x6
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#128x3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 3 * 6)

    def get_model(self):
        return self.model

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(self.relu5(self.conv5(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        classes = torch.softmax(x[:,:9], dim=1)
        x1 = torch.sigmoid(x[:,9:12])
        x2 = torch.sigmoid(x[:,12:15])
        x3 = torch.sigmoid(x[:,15:18])
        return torch.cat((x1, x2, x3, classes), dim=1)