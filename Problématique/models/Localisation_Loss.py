import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch import classes
from torch.nn.functional import mse_loss


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

        # presence_loss = torch.zeros([target.size(0),target.size(1),output.size(1)])
        # mse_loss = torch.zeros([target.size(0),target.size(1),output.size(1)])
        # class_loss = torch.zeros([target.size(0),target.size(1),output.size(1)])
        # for i in range(target.size(1)):
        #     for j in range(output.size(1)):
        #         # Calcul de la perte pour la présente d'un carré
        #         presence_loss[:,i,j] = self.BCELoss(output[:,j, 0], target[:,i, 0])
        #         # Calcul de la perte de localisation
        #         mse_loss[:,i,j] =  self.MSELoss(output[:,j,1:4], target[:,i,1:4])
        #         # Calcul de la perte de classification
        #         class_loss[:,i,j] = self.CrossEntropyLoss(output[:,j,4:], target[:,i,4].long())
        # presence_loss_index = torch.argmin(presence_loss,dim=2)
        # mse_loss_index = torch.argmin(mse_loss,dim=2)
        # class_loss_index = torch.argmin(class_loss,dim=2)
        #
        # return  (self._alpha * presence_loss_min) + (self._beta * mse_loss_min) + (self._gamma * class_loss_min)
        # #return (self._alpha * presence_loss) + (self._beta * MSEloss) + (self._gamma * class_loss)

        loss = 0

        for i in range(target.size(0)):

            predictions = output[i]
            targets = target[i]

            box_predictions = predictions[:, 1:4]
            box_targets = targets[:, 1:4]

            distance = torch.cdist(box_predictions, box_targets, p=2)

            row, col = linear_sum_assignment(distance.detach().cpu().numpy())

            for j,k in zip(row, col):
                mse_loss = self.MSELoss(box_predictions[j], box_targets[k])

                presence_loss = self.BCELoss(predictions[k, 0], targets[j, 0])

                class_prob = predictions[j, 4:]
                class_target = targets[k, 4].long()
                classification_loss = self.CrossEntropyLoss(class_prob.unsqueeze(0), class_target.unsqueeze(0))

                loss += (self._alpha * presence_loss) + (self._beta * mse_loss) + (self._gamma * classification_loss)

        return loss / target.size(0)

