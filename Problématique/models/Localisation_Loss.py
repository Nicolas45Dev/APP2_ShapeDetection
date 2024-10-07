import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.JpegPresets import presets
from networkx.algorithms.shortest_paths.unweighted import predecessor
from scipy.optimize import linear_sum_assignment


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

        # loss = 0
        #
        # for element in range(target.size(0)):
        #     prediction = torch.clone(output[element]) # 3x7
        #     el_target = target[element]
        #
        #     used_index = torch.zeros(prediction.size(0), dtype=torch.bool)
        #
        #     for i in range(el_target.size(0)):
        #         distance = torch.zeros([3])
        #         for j in range(prediction.size(0)):
        #
        #             one_hot = F.one_hot(el_target[i, 4].long(), num_classes=3).float()
        #             result = torch.zeros(prediction[j].size())
        #             result[:4] = el_target[i,:4]
        #             result[4:] = one_hot
        #             distance[j] = torch.norm(prediction[j].detach().cpu() - result.detach().cpu())
        #
        #         index = torch.argmin(distance)
        #
        #         if not used_index[index] and index != i:
        #             inter_pred = torch.clone(prediction[i])
        #             prediction[i] = prediction[index]
        #             prediction[index] = torch.clone(inter_pred)
        #
        #             used_index[index] = True
        #
        #     for i in range(el_target.size(0)):
        #         if el_target[i,0] == 0:
        #             presence_loss = self.BCELoss(prediction[i, 0], el_target[i, 0])
        #             mse_loss = 0
        #             class_loss = 0
        #         else:
        #             presence_loss = self.BCELoss(prediction[i, 0], el_target[i, 0])
        #             mse_loss = self.MSELoss(prediction[i, 1:4], el_target[i, 1:4])
        #             class_loss = self.CrossEntropyLoss(prediction[i, 4:], el_target[i, 4].long())
        #
        #         loss += self._alpha * presence_loss + self._beta * mse_loss + self._gamma *  class_loss
        #
        # return loss / (target.size(0) * target.size(1))

        # loss = nn.MSELoss()  # nn.CrossEntropyLoss()       loss_entropy = nn.CrossEntropyLoss()
        # return loss(output[:, :4], target[:, :, :4]) + self._alpha * loss_entropy(output[:, 4:], c.long())

        # presence_loss = torch.zeros([target.size(0),target.size(1),output.size(1)])
        # mse_loss = torch.zeros([target.size(0),target.size(1),output.size(1)])
        # class_loss = torch.zeros([target.size(0),target.size(1),output.size(1)])
        # loss = torch.zeros([target.size(0), target.size(1), output.size(1)])
        # for i in range(target.size(1)):
        #     for j in range(output.size(1)):
        #         # Calcul de la perte pour la présente d'un carré
        #         presence_loss[:, i, j] = self.BCELoss(output[:, j, 0], target[:, i, 0])
        #         # Calcul de la perte de localisation
        #         mse_loss[:, i, j] = self.MSELoss(output[:, j, 1:4], target[:, i, 1:4])
        #         # Calcul de la perte de classification
        #         class_loss[:, i, j] = self.CrossEntropyLoss(output[:, j, 4:], target[:, i, 4].long())
        #         loss = 0.3 * presence_loss + mse_loss + class_loss
        # loss_index = torch.argmin(loss,dim=2)
        # presence_loss_index = torch.argmin(presence_loss,dim=2)
        # mse_loss_index = torch.argmin(mse_loss,dim=2)
        # class_loss_index = torch.argmin(class_loss,dim=2)
        #
        #
        # return (self._alpha * presence_loss) + (self._beta * MSEloss) + (self._gamma * class_loss)

        loss = 0

        for element in range(target.size(0)):
            predictions = output[element]
            targets = target[element]

            predictions_class = predictions[:, 4:]
            targets_class = F.one_hot(targets[:, 4].long(), num_classes=3).float()

            pred_confiance = predictions[:, 0]
            target_confiance = targets[:, 0]

            # Coordonnées des boîtes englobantes prédites
            pred_boxes = predictions[:, 1:4]
            # Coordonnées des boîtes englobantes cibles
            target_boxes = targets[:, 1:4]

            distance = torch.cdist(pred_boxes, target_boxes, p=2)
            class_distance = torch.cdist(predictions_class, targets_class, p=2)
            presence_distance = torch.cdist(predictions[:, 0].unsqueeze(1), targets[:, 0].unsqueeze(1), p=2)

            presence_mask = target_confiance > 0
            total_distance = distance + (self._alpha * class_distance) + (self._beta * presence_distance)

            row, col = linear_sum_assignment(total_distance.detach().cpu().numpy())

            for i, j in zip(row, col):
                if presence_mask[j]:
                    mse = self.MSELoss(pred_boxes[i], target_boxes[j])
                    presence_loss = self.BCELoss(predictions[i, 0], targets[j, 0])

                    class_probs = predictions[i, 4:]
                    class_target = targets[j, 4].long()
                    class_loss = self.CrossEntropyLoss(class_probs.unsqueeze(0), class_target.unsqueeze(0))

                    loss += mse + presence_loss + class_loss
                else:
                    presence_loss = self.BCELoss(pred_confiance, target_confiance)

                    loss += presence_loss

        return loss / target.size(0)