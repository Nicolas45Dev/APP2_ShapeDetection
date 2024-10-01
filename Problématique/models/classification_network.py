import torch

class ClassificationModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes

        # Definir l'architecture du réseau basé sur LeNet-5
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(), # 32 x 1600 tensor size
            torch.nn.Linear(16 * 10 * 10, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, num_classes),
            torch.nn.ReLU()
        )

        # # Definir l'architecture du réseau basé sur MobileNetV2
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        # self.model.classifier[1] = torch.nn.Linear(1280, num_classes)
        # self.model = self.model.cuda()

    def get_model(self):
        return self.model

    def forward(self, data):
        data = self.model(data)
        return data
