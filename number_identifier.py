# main function to import and identify numbers 1-9 from the MNIST dataset image using modules.py and dataloader.py
import torch
import torch.nn as nn
import torch.optim as optim
import modules
import dataloader

def main():
    # define the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the hyperparameters
    hidden_size:int = 10
    num_classes = 10
    num_epochs = 7
    batch_size = 100
    learning_rate = 0.001
    val_split = 0.2

    # define the MNIST dataset
    mnist_dataset = dataloader.MNISTDataset(batch_size, val_split) # return value is a custom dataset class object

    # print dataset shape for verification
    examples = iter(mnist_dataset.train_loader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)
    # torch.Size([100, 1, 28, 28]) torch.Size([100]) 
    # 100 samples, 1 channel(no colour channels), 28x28 pixels

    # using plt.imshow() to display the first 6 images in the dataset
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(samples[i][0], cmap='gray')
    # # plt.show()

    # define the model
    model = modules.CrossEntropyLossNN(mnist_dataset.get_input_size(), hidden_size, num_classes).to(device)

    # define the loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # define and call the trainer class and related methods
    trainer = modules.Trainer(model, loss, optimizer, mnist_dataset.train_loader, mnist_dataset.test_loader, num_epochs, batch_size, device)
    trainer.train()



if __name__ == '__main__':
    main()
