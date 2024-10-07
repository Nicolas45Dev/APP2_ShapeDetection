import torch
import torch.nn as nn
import torch.nn.functional as F
from triton.language import tensor

class DetectionModel(torch.nn.Module):
    def __init__(self, num_classes, S=6, B=2):
        super(DetectionModel, self).__init__()
        self.num_classes = num_classes
        # Convolutions pour extraire les caractéristiques
        self.feature_extractor = nn.Sequential(# 1x53x53
            nn.Conv2d(1, 8, 3, padding=1),# 8x53x53
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),# 16x26x26
            nn.Conv2d(8, 32, 3, padding=1),# 32x26x26
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),# 32x13x13
            nn.Conv2d(32, 48, 2, padding=0),# 48x12x12
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 32, 3, padding=1),# 32x12x12
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 48, 3, padding=1),# 48x12x12
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3, padding=1),# 48x12x12
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 48, 3, padding=1),# 32x12x12
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)# 32x6x6
        )

        # Couche pour prédire l'existence, la position et la taille des boîtes englobantes
        # On prédit une boîte par cellule
        self.prediction_layer = nn.Sequential(# 32x6x6
            nn.Conv2d(48, 42, 3, padding=0),# 36x4x4
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(42),
            nn.Conv2d(42, 3 * 7 * 7, 4, padding=0),# 147x1x1
            nn.LeakyReLU(0.1),
            nn.Conv2d(3 * 7 * 7, 3 * 7 * 7, 3, padding=1),# 147x1x1 linéaire!
        )# nn.Conv2d(128, 3 * 7, 4, padding=2)


    def get_model(self):
        return self.model

    def forward(self, x):
        features = self.feature_extractor(x)

        predictions = self.prediction_layer(features)

        # Reshape pour le target de la détection (N, 3, 5, S, S)
        N, C, H, W = predictions.shape
        predictions = predictions.view(N, self.num_classes, 7, -1)

        # Combinaison du w et h pour obtenir un wh, le système devra apprendre que w = h
        predictions = torch.mean(predictions, dim=3)

        predictions = torch.sigmoid(predictions)

        return predictions