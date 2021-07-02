import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import datasets, transforms


def plot_aug(plot_misclassified, data, ncol=6):
  nrow = len(aug_dict)

  fig, axes = plt.subplots(ncol, nrow, figsize=( 3*nrow, 15), squeeze=False)
  for i, (key, aug) in enumerate(aug_dict.items()):
    for j in range(ncol):
      ax = axes[j,i]
      if j == 0:
        ax.text(0.5, 0.5, key, horizontalalignment='center', verticalalignment='center', fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
      else:
        image, label = data[j-1]
        if aug is not None:
          transform = A.Compose([aug])
          image = np.array(image)
          image = transform(image=image)['image']
          
        ax.imshow(image)
        ax.set_title(f'{data.classes[label]}')
        ax.axis('off')

  plt.tight_layout()
  plt.show()


class LoadDataset(Dataset):
  def __init__(self, data, transform):
    self.data = data
    self.aug = transform
        
  def __len__(self):
    return (len(self.data))

  def __getitem__(self, i):
      
    image, label = self.data[i]
    
    #apply augmentation only for training
    image = self.aug(image=np.array(image))['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    return torch.tensor(image, dtype=torch.float), label
    

def plot_data(data, rows, cols, lower_value, upper_value):

    figure = plt.figure(figsize=(cols*2,rows*3))
    for i in range(1, cols*rows + 1):
        k = np.random.randint(lower_value,upper_value)
        figure.add_subplot(rows, cols, i) # adding sub plot

        img, label = data.dataset[k]
        
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Class: {label}')

    plt.tight_layout()
    plt.show()


def plot_misclassified(model, test_loader, classes, device, dataset_mean, dataset_std, no_misclf=20, return_misclf=False):
  count = 0
  k = 30
  images = []
  misclf = list()
  labels = []
  
  while count<no_misclf:
    img, label = test_loader.dataset[k]
    pred = model(img.unsqueeze(0).to(device)) # Prediction
    # pred = model(img.unsqueeze(0).to(device)) # Prediction
    pred = pred.argmax().item()

    k += 1
    if pred!=label:
      denormalize = transforms.Normalize((-1 * dataset_mean / dataset_std), (1.0 / dataset_std))
      img = denormalize(img)
      misclf.append([img, label, pred])
      images.append(img)
      labels.append(label)
      count += 1
  
  rows, cols = int(no_misclf/5),5
  figure = plt.figure(figsize=(cols*3,rows*3))

  for i in range(1, cols * rows + 1):
    img, label, pred = misclf[i-1]

    figure.add_subplot(rows, cols, i) # adding sub plot
    plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
    plt.axis("off") # hiding the axis
    img = img.squeeze().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap="gray") # showing the plot

  plt.tight_layout()
  plt.show()
  
  if return_misclf:
    return [images, labels]
  
  
  