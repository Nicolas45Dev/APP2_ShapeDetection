import torch
import torch.nn as nn
import torch.nn.functional as F
from triton.language import tensor

class DetectionModel(torch.nn.Module):
    def __init__(self, num_classes, S=6, B=2):
        super(DetectionModel, self).__init__()
        self.num_classes = num_classes
        # Convolutions pour extraire les caractéristiques
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),  # Change input channel to 1 for grayscale
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, 3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 4, padding=2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        # Couche pour prédire l'existence, la position et la taille des boîtes englobantes
        # On prédit une boîte par cellule
        self.prediction_layer = nn.Sequential(
            nn.Conv2d(128, 96, 3, padding=1),
            nn.Conv2d(96, 3 * 7, 4, padding=2)
        )# nn.Conv2d(128, 3 * 7, 4, padding=2)


    def get_model(self):
        return self.model

    def forward(self, x):
        features = self.feature_extractor(x)

        predictions = self.prediction_layer(features)

        # Reshape pour le target de la détection (N, 3, 5, S, S)
        N, C, H, W = predictions.shape
        predictions = predictions.view(N, self.num_classes, 7, H, W)

        # Combinaison du w et h pour obtenir un wh, le système devra apprendre que w = h
        predictions = torch.mean(predictions, dim=(3, 4))

        predictions = torch.sigmoid(predictions)

        return predictions