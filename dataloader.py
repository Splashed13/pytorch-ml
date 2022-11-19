import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math

# define a custom dataset class for the MNIST dataset that inherits from the Dataset class
# that takes a batch size and a % validation split as arguments
class MNISTDataset(Dataset):
    def __init__(self, batch_size, val_split):
        self.batch_size = batch_size
        self.val_split = val_split
        self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_size = len(self.train_dataset) 
        self.test_size = len(self.test_dataset)
        self.val_size = math.floor(self.train_size * self.val_split)
        self.train_size = self.train_size - self.val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [self.train_size, self.val_size])

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    # a flattened version of the MNIST dataset test_loader
    def get_input_size(self):
        return self.test_dataset[0][0].flatten().shape[0] # will return the integer 784

