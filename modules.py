import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from plotter import Plotter


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# convolutional neural network class for the CIFAR10 dataset
class ConvNet(nn.Module):
    def __init__(self, num_labels, label_names):
        super(ConvNet, self).__init__()
        self.label_names = label_names
        self.num_labels = num_labels
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels, 6 output channels, 5x5 kernel
        # sizing changes: (32 - 5 + 2*0)/1 + 1 = 28  -- due to padding of 0 and the kernel size of 5
        self.pool = nn.MaxPool2d(2, 2) # 2x2 max pooling
        # sizing changes: 28/2 = 14 -- due to the max pooling of 2x2
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel
        # output size is 10x10 -- due to the kernel size of 5 and the input size of 14x14
        # this flattens the 16 5x5 feature maps into a 1D vector of 400 elements
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 120 is the hidden layer size -- can be any number
        self.fc2 = nn.Linear(120, 84)  # 84 is the hidden layer size -- can be any number
        self.fc3 = nn.Linear(84, 10)  

    # this function defines the forward pass of the network it is called automatically when the network is called
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# cross entropy loss NN classifer for multi-class classification with linear hidden layer activated by ReLU
class CrossEntropyLossNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, label_names):
        super(CrossEntropyLossNN, self).__init__()  
        self.input_size = input_size
        self.num_labels = num_labels
        self.label_names = label_names
        self.linear1 = nn.Linear(input_size, hidden_size)
        # try a classification activation function
        self.softmax = nn.LogSoftmax(dim=1) 
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        out = self.linear1(x)
        out = self.softmax(out)
        #out = self.relu(out)
        out = self.linear2(out)
        return out


# trainer class for training a model with loss = nn.CrossEntropyLoss() and optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
class Trainer:
    def __init__(self, model, loss, optimizer, train_loader, test_loader, num_epochs, batch_size, device):
        # fields for plotting the loss and accuracy
        self.batch_size = batch_size
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device

    def train(self):
        print("\n[Starting Training]\n")
        # start plotting thread for loss and accuracy
        p = Plotter()
        n_total_steps = len(self.train_loader) # is calculated as len(train_dataset) / batch_size
        # get number of labels for accuracy
        # total number of batches
        for epoch in range(self.num_epochs):
            train_loss = 0
            batch_smoothed = []
            for i, (images, labels) in enumerate(self.train_loader): # each batch 
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                outputs = self.model(images)
                l = self.loss(outputs, labels)
                train_loss = train_loss + l.item()
                
                # backward pass
                self.optimizer.zero_grad() # we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
                l.backward() # calculate gradients
                self.optimizer.step() # update the weight
        
                # for console output so we can track the progress
                batch_smoothed.append(l.item())
                if (i+1) % (len(self.test_loader)/2) == 0:
                    avg_train_loss_smoothed = sum(batch_smoothed) / len(batch_smoothed)
                    print(f'Training... Epoch {epoch+1}/{self.num_epochs}, Step {i+1}/{n_total_steps}, Training Loss over {int(len(self.test_loader)/2)} batches = {avg_train_loss_smoothed:.4f}')
                    batch_smoothed.clear()
                    # OR plot with this for every half number of test batches 
                    batch_test_loss, batch_test_acc = self.test_acc_loss(type="batch_smoothed") 
                    y = [avg_train_loss_smoothed, batch_test_loss, batch_test_acc]
                    p.plot_batch((i+1)+(epoch*n_total_steps),y)

    
            avg_train_loss = train_loss / n_total_steps # epoch loss
            # calculate test loss and accuracy
            avg_test_epoch_loss, epoch_accuracy = self.test_acc_loss(type="epoch")

            print(f'\n[Completed Epoch {epoch+1}/{self.num_epochs}] Step {i+1}/{n_total_steps} -> Avg Training Loss: {avg_train_loss:.4f}, Avg Test Loss: {avg_test_epoch_loss:.4f}, Model Accuracy: {epoch_accuracy*100:.2f}%\n')
            p.plot_epoch(epoch + 1, avg_train_loss, avg_test_epoch_loss, epoch_accuracy)

        self.print_accuracy()

    # method gets the accuracy for each label and prints them at the end of the training
    def print_accuracy(self):
        # overall accuracy
        correct = [0 for i in range(self.model.num_labels)]
        total = [0 for i in range(self.model.num_labels)]
        with torch.no_grad():
            for images, labels in self.test_loader: # each batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()

                for i in range(len(labels)):
                    label = labels[i]
                    correct[label] += c[i].item()
                    total[label] += 1

        # print new line 
        print()
        for i in (range(self.model.num_labels)):
            print(f'Accuracy of {self.model.label_names[i]}: {(100 * correct[i] / total[i]):0.2f}%')

        print()
        # overall accuracy summing all correct and total
        print(f'Overall Accuracy Model: {(100 * sum(correct) / sum(total)):0.2f}%')

    
    # method that calculates the loss over a random single batch or all batches (if epoch) depending on the input
    def test_acc_loss(self,type=None):
        n_correct = 0
        n_samples = 0
        test_loss = 0
        if type == "epoch":
            with torch.no_grad():
                for k, (images, labels) in enumerate(self.test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    l2 = self.loss(outputs, labels)
                    test_loss = test_loss + l2.item()
                    n_samples = n_samples + labels.shape[0]
                    n_correct = n_correct + (outputs.argmax(1) == labels).sum().item()
            avg_test_loss = test_loss / (k+1)
            avg_accuracy = n_correct / n_samples
            return avg_test_loss, avg_accuracy

        elif type == "batch_smoothed":
            i = 0
            with torch.no_grad():
                # randomly generate batches/2 numbers between 0 and the number of batches for the test set
                random_batch = random.sample(range(0,len(self.test_loader)-1),int(len(self.test_loader)/2))
                for k, (images, labels) in enumerate(self.test_loader):
                    if k in random_batch:
                        i = i + 1
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(images)
                        l2 = self.loss(outputs, labels)
                        test_loss = test_loss + l2.item()
                        n_samples = n_samples + labels.shape[0]
                        n_correct = n_correct + (outputs.argmax(1) == labels).sum().item()
                    else:
                        continue
                avg_test_loss = test_loss / i
                avg_accuracy = n_correct / n_samples
                return avg_test_loss, avg_accuracy

        else:
            print("Invalid type argument. Please use 'epoch' or 'batch'")
            # exit program
            sys.exit()




        


            
           
        



