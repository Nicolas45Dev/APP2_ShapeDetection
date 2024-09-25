import torch.nn as nn
from numpy.ma.core import shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        self.fc1 = nn.Conv2d(1, 4, 3, 1, 1)
        self.fc2 = nn.BatchNorm2d(4)
        self.fc3 = nn.ReLU()
        self.fc4 = nn.MaxPool2d(2,2)
        self.fc5 = nn.Conv2d(4, 2, 3, 1, 1)
        self.fc6 = nn.BatchNorm2d(2)
        self.fc7 = nn.ReLU()
        self.fc8 = nn.MaxPool2d(2,2)
        self.fc9 = nn.Conv2d(2, 10, 7, 1)

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------

    def forward(self, x):
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        output = self.fc6(output)
        output = self.fc7(output)
        output = self.fc8(output)
        output = self.fc9(output)

        output = output.view(x.shape[0], -1)
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------
        return output
