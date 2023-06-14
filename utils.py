import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time


def compute_accuracy(model, data_loader, device):
    """
    Calculates the accuracy of a given data loader.

    Parameters:
        model (nn.Module): the used CNN model
        data_loader (DataLoader): train loader or test loader
        device: the used device
    
    Returns:
        (float): the calculated accuracy
    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    return (correct_pred.float()/num_examples) * 100


def train_model(model: nn.Module, num_epochs: int, train_loader: DataLoader,
                loss_function, optimizer, device=None, logging_interval=50):
    """
    Helper function used for training purposes.

    Parameters:
        model (nn.Module): the used CNN model
        num_epochs (int): number of epochs
        train_loader (DataLoader): the train loader
        loss_function: the loss function that will be used
        optimizer: the used optimizer
        device: the used device
        logging_interval (int): number of minibatches to log after

    Returns:
        minibatch_loss_list (List): list of all calculated losses in the minibatch
        train_acc_list (List): list of all calculated accuracies during the training phase
    """
    start_time = time.time()
    minibatch_loss_list, train_acc_list, = [], []   

    for epoch in range(num_epochs):
        # Putting the model in training mode
        model.train()

        inner_tqdm = tqdm(train_loader, desc=f"Training | Epoch {epoch+1}/{num_epochs} ", leave=True, position=0)

        for batch_idx, (features, targets) in enumerate(inner_tqdm):
            features = features.to(device)
            targets = targets.to(device)

            # Forward && Backward Propagation
            logits = model(features)
            loss = loss_function(logits, targets)
            optimizer.zero_grad()
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # Logging after 50 Batches
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Loss after {batch_idx} Batches : {loss:.4f}')
        
        # Putting model in evalution mode
        model.eval()
        with torch.no_grad():
          train_acc = compute_accuracy(model, train_loader, device=device)
          print(f"Accuracy after {epoch+1} epoch(s) ===> {train_acc:.2f}")
          train_acc_list.append(train_acc.item())


    # Total time
    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    return minibatch_loss_list, train_acc_list


def test_model(model: nn.Module, test_loader: DataLoader, num_epochs, loss_function, optimizer, device=None):
  """
  Helper function used for testing purposes.

    Parameters:
        model (nn.Module): the used CNN model
        test_loader (DataLoader): the test loader
        num_epochs (int): number of epochs
        loss_function: the loss function that will be used
        optimizer: the used optimizer
        device: the used device

    Returns:
        test_acc_list (List): list of all calculated accuracies during the testing phase
  """
  start_time = time.time()
  test_acc_list = []

  for epoch in range(num_epochs):
    # Putting the model in evaluation mode
    model.eval()

    inner_tqdm = tqdm(test_loader, desc=f"Testing | Epoch {epoch+1}/{num_epochs} ", leave=True, position=0)

    for batch_idx, (features, targets) in enumerate(inner_tqdm):
      features = features.to(device)
      targets = targets.to(device)

      # Forward && Backward Propagation
      logits = model(features)
      loss = loss_function(logits, targets)
      optimizer.zero_grad()
      loss.backward()

      # Update model parameters
      optimizer.step()

    with torch.no_grad():
      print("Calculating accuracy ....")
      test_acc = compute_accuracy(model, test_loader, device=device)
      print(f"Accuracy after {epoch+1} epoch(s) ===> {test_acc:.2f} %")
      test_acc_list.append(test_acc.item())

  # Total time
  elapsed = (time.time() - start_time) / 60
  print(f'Total Testing Time: {elapsed:.2f} min')

  return test_acc_list