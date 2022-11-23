import torch
import torch.nn as nn
import torch.optim as optim
from modules import ConvNet, Trainer
import dataloader
import time
import sys
import optuna

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())

# main function to identify the objects in the CIFAR10 dataset using modules.py and dataloader.py
def main():
    start_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the hyperparameters
    num_epochs = 3
    batch_size = 25
    learning_rate = 0.001
    val_split = 0.2

    # define the CIFAR10 dataset
    cifar10_dataset = dataloader.CIFAR10Dataset(batch_size, val_split)

    # define the model
    model = ConvNet(cifar10_dataset.get_num_classes(),cifar10_dataset.get_label_names()).to(device)
    
    loss = nn.CrossEntropyLoss() # CrossEntropyLoss combines LogSoftmax and NLLLoss

    # one of the best optimizers for CNNs is Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    trainer = Trainer(model, loss, optimizer, cifar10_dataset.train_loader, cifar10_dataset.test_loader, num_epochs, batch_size, device)

    trainer.train()

    print(f"\n[Training Complete] Training time: {time.perf_counter() - start_time:0.2f} seconds")
    # gracefully exit the program
    input("Press Enter to Exit...")
    sys.exit()


if __name__ == '__main__':
    main()

