import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import math

# define a custom dataset class for the MNIST dataset that inherits from the Dataset class
# that takes a batch size and a % validation split as arguments
class MNISTDataset(Dataset):
    def __init__(self, batch_size, val_split): 
        # normalize the data so the values are -1 to 1 
        transform_ = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,),(0.5,)), torchvision.transforms.Lambda(lambda x: x.flatten())])
        #transform_ = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]) # the normalization values are from the MNIST dataset documentation
        self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_, download=True)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_, download=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.train_size = len(self.train_dataset) 
        self.test_size = len(self.test_dataset)
        val_size = math.floor(self.train_size * val_split)
        self.train_size = self.train_size - val_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [self.train_size, val_size])

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    # a flattened version of the MNIST dataset test_loader
    def get_input_size(self):
        return self.test_dataset[0][0].flatten().shape[0] # will return the integer 
        
    # get label names in order they would be in the test_loader
    def get_label_names(self):
        return self.test_dataset.classes

    # get the number of classes in the dataset
    def get_num_classes(self):
        return len(self.test_dataset.classes)


class CIFAR10Dataset(Dataset):
    def __init__(self, batch_size, val_split):
        # normalization values are from the CIFAR10 dataset documentation
        transform_ = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = batch_size
        self.val_split = val_split
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_, download=True)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_, download=True)
        # this dataloader takes 30% of the training data set and uses it as a subset for hyperparameter tuning
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.train_size = len(self.train_dataset) 
        self.test_size = len(self.test_dataset)
        self.val_size = math.floor(self.train_size * self.val_split)
        self.train_size = self.train_size - self.val_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [self.train_size, self.val_size])

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    # a flattened version of the CIFAR10 dataset test_loader
    def get_input_size(self):
        return self.test_dataset[0][0].flatten().shape[0] 

    # get label names for CIFAR10 dataset in order they would be in the test_loader
    def get_label_names(self):
        return self.test_dataset.classes  

    def get_num_classes(self):
        return len(self.test_dataset.classes)      