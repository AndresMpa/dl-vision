import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.characteristic = nn.Sequential(
            # Input block
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # Conv 1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # Conv 2
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # Conv 3
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # Conv 4
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Conv 6
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Conv 7
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Conv 8
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.flatter = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )

        self.classificator_nn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x_data):
        value_tracker = self.characteristic(x_data)
        value_tracker = self.flatter(value_tracker)
        value_tracker = self.classificator_nn(value_tracker)

        return value_tracker
