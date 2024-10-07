import numpy as np
import torch
import torch.nn as nn
from sympy.physics.units import length
from tensorflow.dtensor.python.config import is_gpu_present


class LocalizationLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(LocalizationLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        self.BCELoss = nn.BCELoss()
        self.MSELoss = nn.MSELoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        # Output se compose de 4 colonnes : x, y, w, h, une prédiction de la boite dans l'image (N, 4)
        # Target se compose de 5 colonnes : x, y, w, h, c pour chaque boîte de forme (N, 3, 5)
        # Somme la différence entre les coordonnées prédites et les coordonnées réelles

        # Calcul de la perte pour la présente d'un carré
        presence_loss = self.BCELoss(output[:,:, 0], target[:,:, 0])
        # Calcul de la perte de localisation
        MSEloss =  self.MSELoss(output[:,:,:4], target[:,:,:4])
        # Calcul de la perte de classification
        class_loss = self.CrossEntropyLoss(output[:,:,4:], target[:,:,4].long())
        # y = 1 / (self._alpha + self._beta + self._gamma)

        return  self._beta * MSEloss + self._gamma * class_loss
