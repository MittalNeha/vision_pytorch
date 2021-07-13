import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImagenetDataset(Dataset):
    '''
    get the image and label from tiny_imagenet dataset
    '''

    def __init__(self, path, transforms=None):
        self.transform = transforms
        self.image_list, self.labels = self.get_data(path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)
        return image, label

    def get_id_dictionary(self, path):
        id_dict = {}
        for i, line in enumerate(open(path + 'wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict

    def get_data(self, path):
        print('starting loading data')
        data = []
        labels = []
        for key, value in self.get_id_dictionary(path).items():
            data += [path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)) for i in range(500)]
            labels_ = np.array([value] * 500)
            labels += labels_.tolist()
        return np.array(data), np.array(labels)