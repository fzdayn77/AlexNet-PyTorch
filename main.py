import os
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from model import AlexNet
from utils import train_model, test_model


# Hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 200
RANDOM_SEED = 123
NUM_CLASSES = 10 # Number of classes in the CIFAR-10 Dataset
NUM_WORKERS = os.cpu_count()
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# Transformations
train_transforms = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomCrop((65, 65)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
])

test_transforms = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.CenterCrop((65, 65)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
])


# Path
cifar10_path = './data/CIFAR-10'

# Trainset and Trainloader
train_set = CIFAR10(root=cifar10_path, train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)

# Testset and Testloader
test_set = CIFAR10(root=cifar10_path, train=False, download=True, transform=test_transforms)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)

# AlexNet model
model = AlexNet(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Optimizer
optimizer = SGD(model.parameters(), momentum=0.9, lr=0.01)

# Loss Function
loss_function = nn.CrossEntropyLoss()

# Training
minibatch_loss_list, train_acc_list = train_model(model=model, num_epochs=NUM_EPOCHS, train_loader=train_loader,
                                                  loss_function=loss_function, optimizer=optimizer, device=DEVICE)

# Testing
test_acc_list = test_model(model=model, test_loader=test_loader, num_epochs=NUM_EPOCHS,
                            loss_function=loss_function, optimizer=optimizer, device=DEVICE)
