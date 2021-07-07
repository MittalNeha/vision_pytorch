from tqdm.notebook import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn


def download_data():
  trainset = torchvision.datasets.CIFAR10(root='/content/data', train=True,
                                          download=True )
  testset = torchvision.datasets.CIFAR10(root='/content/data', train=False,
                                        download=True)
  return trainset,testset
  

def train(model, device, train_loader, optimizer, use_l1=False, lambda_l1=0.01, scheduler=None):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss = 0
  hist_lr = []
  
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch 
    # accumulates the gradients on subsequent backward passes. Because of this, when you start your training loop, 
    # ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss

    loss = nn.CrossEntropyLoss()(y_pred, target)

    l1=0
    if use_l1:
      for p in model.parameter():
        l1 = l1 + p.abs().sum()
    
    loss = loss + lambda_l1*l1

    # Backpropagation
    loss.backward()
    optimizer.step()

    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()

    train_loss += loss.item()
    
    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    hist_lr.append(scheduler.get_last_lr())

    pbar.set_description(desc= f'Batch_id={batch_idx} Loss={train_loss/(batch_idx + 1):.5f} Accuracy={100*correct/processed:0.2f}')

  return 100*correct/processed, train_loss/(batch_idx + 1), hist_lr

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    iteration = len(test_loader.dataset)// test_loader.batch_size
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= iteration

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    
    return 100. * correct / len(test_loader.dataset), test_loss

def fit_model(net, scheduler, optimizer, device, NUM_EPOCHS,train_loader, test_loader, use_l1=False):
  training_acc, training_loss, testing_acc, testing_loss = list(), list(), list(), list()
  train_lr = list()
  
  for epoch in range(1,NUM_EPOCHS+1):
      print("EPOCH:", epoch)
      train_acc, train_loss, lr = train(model=net, device=device, train_loader=train_loader, optimizer=optimizer, use_l1=use_l1, scheduler=scheduler)
      test_acc, test_loss = test(net, device, test_loader)
      # update LR
      if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step(test_loss)
      training_acc.append(train_acc)
      training_loss.append(train_loss)
      testing_acc.append(test_acc)
      testing_loss.append(test_loss)
      train_lr.extend(lr)
      
  return net, (training_acc, training_loss, testing_acc, testing_loss, train_lr)

