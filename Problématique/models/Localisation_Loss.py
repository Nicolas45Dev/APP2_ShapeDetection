from inspect import stack

import numpy as np
import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self, alpha):
        super(LocalizationLoss, self).__init__()
        self._alpha = alpha

    def forward(self, output, target):
        # Output se compose de 4 colonnes : x, y, w, h, une prédiction de la boite dans l'image (N, 4)
        # Target se compose de 5 colonnes : x, y, w, h, c pour chaque boîte de forme (N, 3, 5)
        # Somme la différence entre les coordonnées prédites et les coordonnées réelles
        x1 = output[:,:4]
        x2 = output[:,4:8]
        x3 = output[:,8:12]
        classes1 = output[:, 12:15]
        classes2 = output[:, 15:18]
        classes3 = output[:, 18:21]
        classes = torch.stack([classes1, classes2, classes3], dim=1)
        c = target[:,:,4:]
        result1 = torch.stack([x1, x2, x3], dim=1)
        loss = nn.MSELoss()
        loss_entropy = nn.CrossEntropyLoss()
        mse = loss(result1[:,:,0:3], target[:,:,1:4])
        entropy = loss_entropy(classes, c)
        return loss(result1[:,:,0:3], target[:,:,1:4]) + self._alpha * loss_entropy(classes, c.long())

def check_loss_output_size(loss):
    assert loss.size() == torch.Size([]), f'La sortie de la fonction de coût ({loss.item()}) doit être un scalaire.'


def check_loss_output_near(loss, expect_loss):
    assert abs(loss.item() - expect_loss) < 1e-3, f'La sortie de la fonction de coût ({loss.item()}) n\'est pas proche de {expect_loss}.'


def check_loss_output_smaller_than(loss1, loss2):
    assert loss1.item() < loss2.item(), f'La sortie de la fonction de coût ({loss1.item()}) n\'est pas plus petite que de {loss2.item()}.'


def test_reduction():
    print("Test - Reduction")
    criterion = LocalizationLoss(alpha=2)
    output = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                           [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
    target = torch.tensor([[0.4, 0.35, 0.6, 0.65, 1.0],
                           [0.4, 0.35, 0.6, 0.65, 0.0]])

    loss = criterion(output, target)
    check_loss_output_size(loss)
    print("\tOk")


def test_perfect_output():
    print("Test - Perfect output")
    criterion = LocalizationLoss(alpha=2)

    output = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    target = torch.tensor([[0.4, 0.35, 0.6, 0.65, 2.0]])

    loss = criterion(output, target)
    check_loss_output_near(loss, 0.0)
    print("\tOk")


def test_alpha_class_ok():
    print("Test - Alpha when the class is ok")
    criterion_big_alpha = LocalizationLoss(alpha=2)
    criterion_small_alpha = LocalizationLoss(alpha=1)

    output = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10]])
    target = torch.tensor([[0.1, 0.2, 0.3, 0.4, 2.0]])

    loss_big_alpha = criterion_big_alpha(output, target)
    loss_small_alpha = criterion_small_alpha(output, target)
    check_loss_output_near(loss_big_alpha, loss_small_alpha.item())
    print("\tOk")


def test_alpha_class_not_ok():
    print("Test - Alpha when the class is not ok")
    criterion_big_alpha = LocalizationLoss(alpha=2)
    criterion_small_alpha = LocalizationLoss(alpha=1)

    output = torch.tensor([[0.1, 0.3, 0.2, 0.4, 0.0, 1e3, 0.0]])
    target = torch.tensor([[0.0, 0.1, 0.2, 0.5, 0.0]])

    loss_big_alpha = criterion_big_alpha(output, target)
    loss_small_alpha = criterion_small_alpha(output, target)
    check_loss_output_near(loss_big_alpha, 2 * loss_small_alpha.item())
    print("\tOk")


def test_box():
    print("Test - Box")
    criterion = LocalizationLoss(alpha=2)

    output_perfect = torch.tensor([[0.5, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_x_not_perfect = torch.tensor([[0.52, 0.5, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_y_not_perfect = torch.tensor([[0.5, 0.4, 0.2, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_w_not_perfect = torch.tensor([[0.5, 0.5, 0.1, 0.3, 0.0, 0.0, 1e10, 0.0]])
    output_h_not_perfect = torch.tensor([[0.5, 0.5, 0.2, 0.5, 0.0, 0.0, 1e10, 0.0]])
    target = torch.tensor([[0.4, 0.35, 0.6, 0.65, 2.0]])

    loss_perfect = criterion(output_perfect, target)
    check_loss_output_smaller_than(loss_perfect, criterion(output_x_not_perfect, target))
    check_loss_output_smaller_than(loss_perfect, criterion(output_y_not_perfect, target))
    check_loss_output_smaller_than(loss_perfect, criterion(output_w_not_perfect, target))
    check_loss_output_smaller_than(loss_perfect, criterion(output_h_not_perfect, target))
    print("\tOk")


if __name__ == '__main__':
    test_reduction()
    test_perfect_output()
    test_alpha_class_ok()
    test_alpha_class_not_ok()
    test_box()
