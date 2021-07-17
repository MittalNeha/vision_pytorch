import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader


def download_data(path='/content/data'):
    """Download CIFAR10 dataset form torchvision datasets

    Args:
        path (str, optional): download path for dataset. Defaults to '/content/data'. (For Google Colab)

    Returns:
        torchvision instance: training and testing set
    """
    trainset = CIFAR10(root=path, train=True, download=True )
    testset = CIFAR10(root=path, train=False, download=True)

    print('Downloaded CIFAR10 to', path)
    return trainset,testset


class LoadDataset(Dataset):
    """Torch Dataset instance for loadinf dataset and transforming it
    """
    def __init__(self, data, transform):
        self.data = data
        self.aug = transform
    
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, i):
        """Read image from dataset and performs transformations over its

        Args:
            i (int): index of the sample to be accessed

        Returns:
            tensor: tansformed image and label
        """
        image, label = self.data[i]
        
        #apply augmentation only for training
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return torch.tensor(image, dtype=torch.float), label


def get_train_test_loaders(train_transforms, test_transforms, BATCH_SIZE):
    """Generate Torch instance for Train and Test data loaders

    Args:
        train_transforms (albumentations compose class): training tansformations to be applied over images
        test_transforms (albumentations compose class): testing tansformations to be applied over images
        BATCH_SIZE (int): Batch size to be used

    Returns:
        torch instace: train and test data loaders
    """
    trainset, testset = download_data()

    train_loader = DataLoader(LoadDataset(trainset, train_transforms), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
    test_loader = DataLoader(LoadDataset(testset, test_transforms), batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=1)
    return train_loader, test_loader


