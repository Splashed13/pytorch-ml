# main function to import and identify numbers 1-9 from the MNIST dataset image using modules.py and dataloader.py
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import modules
import dataloader
import time

def main():
    # time the training
    start_time = time.perf_counter()
    # define the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the hyperparameters
    hidden_size = 10
    num_epochs = 3
    batch_size = 100
    learning_rate = 0.001
    val_split = 0.2

    # define the MNIST dataset
    mnist_dataset = dataloader.MNISTDataset(batch_size, val_split) # return value is a custom dataset class object

    model = modules.CrossEntropyLossNN(mnist_dataset.get_input_size(), hidden_size, mnist_dataset.get_num_classes(), mnist_dataset.get_label_names()).to(device)

    # define the loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001) # weight decay is L2 regularization


    trainer = modules.Trainer(model, loss, optimizer, mnist_dataset.train_loader, mnist_dataset.test_loader, num_epochs, batch_size, device)
    trainer.train()

    print(f"Training time: {time.perf_counter() - start_time:0.2f} seconds")
    # gracefully exit the program
    input("Press Enter to Exit...")
    sys.exit()


if __name__ == '__main__':
    main()
