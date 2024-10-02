import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        # ------------------------ Laboratoire 2 - Question 5 - Début de la section à compléter ------------------------

        # Down 1
        self.conv_1_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=1)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1)
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)#24
        self.conv_2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#12
        self.conv_3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)#6
        self.conv_4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.relu_4_2 = nn.ReLU() # Sortie vers Concate

        # Down 5
        # self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv_5_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        # self.relu_5_1 = nn.ReLU()
        # self.conv_5_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)
        # self.relu_5_2 = nn.ReLU()

        # Up 6
        # self.upsample_6 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        self.conv_6_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)
        self.relu_6_1 = nn.ReLU()
        self.conv_6_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)#12
        self.conv_7_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.relu_7_2 = nn.ReLU()

        # Up 8
        self.upsample_8 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)#24
        self.conv_8_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.relu_8_1 = nn.ReLU()
        self.conv_8_2 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.relu_8_2 = nn.ReLU()

        # Up 9
        self.upsample_9 = nn.ConvTranspose2d(16, 16, kernel_size=6, stride=2, padding=0,output_padding=1) #24 -> 53
        self.conv_9_1 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.relu_9_1 = nn.ReLU()
        self.conv_9_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.relu_9_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # Down 1
        x = self.conv_1_1(x)#53x53
        x = self.relu_1_1(x)
        x = self.conv_1_2(x)
        x = self.relu_1_2(x)
        self.hidden_h1 = x

        # Down 2
        x = self.maxpool_2(x)#24
        x = self.conv_2_1(x)
        x = self.relu_2_1(x)
        x = self.conv_2_2(x)
        x = self.relu_2_2(x)
        self.hidden_h2 = x

        # Down 3
        x = self.maxpool_3(x)#12
        x = self.conv_3_1(x)
        x = self.relu_3_1(x)
        x = self.conv_3_2(x)
        x = self.relu_3_2(x)
        self.hidden_h3 = x

        # Down 4
        x = self.maxpool_4(x)#6
        x = self.conv_4_1(x)
        x = self.relu_4_1(x)
        x = self.conv_4_2(x)
        x = self.relu_4_2(x)
        # self.hidden_h4 = x

        # Down 5
        # x = self.maxpool_5(x)
        # x = self.conv_5_1(x)
        # x = self.relu_5_1(x)
        # x = self.conv_5_2(x)
        # x = self.relu_5_2(x)

        # Up 6
        # x = self.upsample_6(x)
        # x = self.conv_6_1(torch.cat([x,self.hidden_h4], dim=1))
        x = self.conv_6_1(torch.cat([x, x], dim=1))#13
        x = self.relu_6_1(x)
        x = self.conv_6_2(x)
        x = self.relu_6_2(x)

        # Up 7
        x = self.upsample_7(x)
        x = self.conv_7_1(torch.cat([x, self.hidden_h3], dim=1))
        x = self.relu_7_1(x)
        x = self.conv_7_2(x)
        x = self.relu_7_2(x)

        # Up 8
        x = self.upsample_8(x)
        x = self.conv_8_1(torch.cat([x, self.hidden_h2], dim=1))
        x = self.relu_8_1(x)
        x = self.conv_8_2(x)
        x = self.relu_8_2(x)

        # Up 9
        x = self.upsample_9(x)
        x = self.conv_9_1(torch.cat([x, self.hidden_h1], dim=1))
        x = self.relu_9_1(x)
        x = self.conv_9_2(x)
        x = self.relu_9_2(x)

        x = self.output_conv(x)
        # x = torch.sigmoid(x)

        return x
        # ------------------------ Laboratoire 2 - Question 5 - Fin de la section à compléter --------------------------
