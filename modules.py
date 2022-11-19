import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from plotter import Plotter

# cross entropy loss NN classifer for multi-class classification with linear hidden layer activated by ReLU
class CrossEntropyLossNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CrossEntropyLossNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # try a classification activation function
        self.softmax = nn.LogSoftmax(dim=1) 
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

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
        # start plotting thread for loss and accuracy
        p = Plotter()
        window_size = 100
        n_total_steps = len(self.train_loader) # is calculated as len(train_dataset) / batch_size
        # total number of batches
        for epoch in range(self.num_epochs):
            train_loss = 0
            window_training_loss = []
            for i, (images, labels) in enumerate(self.train_loader): # each batch 
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(self.device)
                labels = labels.to(self.device)

                # forward pass
                outputs = self.model(images)
                l = self.loss(outputs, labels)
                train_loss = train_loss + l.item()
                
                # backward pass
                self.optimizer.zero_grad()
                l.backward() # calculate gradients
                self.optimizer.step() # update the weights

                # for console output so we can track the progress
                window_training_loss.append(l.item())
                if (i+1) % (len(self.test_loader)+1) == 0:
                    avg_train_loss_smoothed = sum(window_training_loss) / (len(self.test_loader)+1)
                    print(f'Training... Epoch {epoch+1}/{self.num_epochs}, Step {i+1}/{n_total_steps}, Training Loss over 100 batches = {avg_train_loss_smoothed:.4f}')
                    window_training_loss.clear()
                    # OR plot with this for every half number of test batches 
                    batch_test_acc, batch_test_loss = self.test_acc_loss(type="batch_smoothed")
                    y = [avg_train_loss_smoothed, batch_test_loss, batch_test_acc]
                    p.plot_batch((i+1)+((epoch+1)*n_total_steps),y)
                
                # plot loss over smoothness amount of steps (batches)
                #p.plot_batch((i+1)+((epoch+1)*n_total_steps),l.item(),smoothness=20)

                # or plot over each batch
                # batch_test_acc, batch_test_loss = self.test_acc_loss(type="batch")
                # y = [l.item(), batch_test_acc, batch_test_loss]
                # p.plot_batch((i+1)+((epoch+1)*n_total_steps),y)

            avg_train_loss = train_loss / n_total_steps # epoch loss
            # calculate test loss and accuracy
            avg_test_epoch_loss, epoch_accuracy = self.test_acc_loss(type="epoch")

            print(f'[Completed Epoch {epoch+1}/{self.num_epochs}] Step {i+1}/{n_total_steps} -> Avg Training Loss: {avg_train_loss:.4f}, Avg Test Loss: {avg_test_epoch_loss:.4f}, Model Accuracy: {epoch_accuracy*100:.2f}%')
            p.plot_epoch(epoch + 1, avg_train_loss, avg_test_epoch_loss, epoch_accuracy)

        input("Press Enter to Finish...")
        sys.exit()
    
    # method that calculates the loss over a random single batch or all batches (if epoch) depending on the input
    def test_acc_loss(self,type=None):
        n_correct = 0
        n_samples = 0
        test_loss = 0
        if type == "epoch":
            with torch.no_grad():

                for k, (images, labels) in enumerate(self.test_loader):
                    images = images.reshape(-1, 28*28).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    l2 = self.loss(outputs, labels)
                    test_loss = test_loss + l2.item()
                    n_samples = n_samples + labels.shape[0]
                    n_correct = n_correct + (outputs.argmax(1) == labels).sum().item()
            avg_test_loss = test_loss / (k+1)
            avg_accuracy = n_correct / n_samples
            return avg_test_loss, avg_accuracy
        
        # calculate loss and accuracy over a random batch
        elif type == "batch":
            with torch.no_grad():
                # randomly number between 0 and the number of batches for the test set
                random_batch = random.randint(0,len(self.test_loader)-1)
                for k, (images, labels) in enumerate(self.test_loader):
                    if random_batch == k:
                        images = images.reshape(-1, 28*28).to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(images)
                        l2 = self.loss(outputs, labels)
                        test_loss = l2.item()
                        n_samples = labels.shape[0]
                        n_correct = (outputs.argmax(1) == labels).sum().item()
                    else:
                        continue
                accuracy = n_correct / n_samples
                return test_loss, accuracy

        elif type == "batch_smoothed":
            i = 0
            with torch.no_grad():
                # randomly generate batches/2 numbers between 0 and the number of batches for the test set
                random_batch = random.sample(range(0,len(self.test_loader)-1),int(len(self.test_loader)/2))
                for k, (images, labels) in enumerate(self.test_loader):
                    if k in random_batch:
                        i = i + 1
                        images = images.reshape(-1, 28*28).to(self.device)
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






        


            
           
        



