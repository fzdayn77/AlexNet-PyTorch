import torch
import torch.nn as nn


class AlexNet(nn.Module):
  """
  Implementation from scratch of the AlexNet-model described in https://dl.acm.org/doi/pdf/10.1145/3065386
  with some changes so that it works on the CIFAR-10 dataset.
  """
  def __init__(self, num_classes: int):
      super().__init__()
      self.num_classes = num_classes

      # Features Extraction
      self.features = torch.nn.Sequential(
              # Conv 1
              torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
              torch.nn.ReLU(inplace=True),
              torch.nn.MaxPool2d(kernel_size=3, stride=2),
              # Conv 2
              torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
              torch.nn.ReLU(inplace=True),
              torch.nn.MaxPool2d(kernel_size=3, stride=2),
              # Conv 3
              torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
              torch.nn.ReLU(inplace=True),
              # Conv 4
              torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
              torch.nn.ReLU(inplace=True),
              # Conv 5
              torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
              torch.nn.ReLU(inplace=True),
              torch.nn.MaxPool2d(kernel_size=3, stride=2),
          )
      
      # Average Pooling
      self.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))

      # Classification
      self.classifier = torch.nn.Sequential(
          # Fc 1
          torch.nn.Dropout(0.5),
          torch.nn.Linear(256*4*4, 4096),
          torch.nn.ReLU(inplace=True),
          # Fc 2
          torch.nn.Dropout(0.5),
          torch.nn.Linear(4096, 4096),
          torch.nn.ReLU(inplace=True),
          # Output layer
          torch.nn.Linear(4096, self.num_classes)
      )

  def forward(self, x):
      x = self.features(x)
      x = self.avgpool(x)
      x = torch.flatten(x, start_dim=1)
      logits = self.classifier(x)
      return logits