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
            test_loss = 0
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
                #p.plot_batch(i+1,l.item())
                window_training_loss.append(l.item())
                
                # backward pass
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step() # update the weights

                if (i+1) % window_size == 0:
                    print(f'Training... Epoch {epoch+1}/{self.num_epochs}, Step {i+1}/{n_total_steps}, Training Loss over 100 batchs = {sum(window_training_loss) / window_size:.4f}')
                    window_training_loss.clear()
                
            

            # evaluate over the entire test set for each epoch
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for k, (images, labels) in enumerate(self.test_loader):
                    images = images.reshape(-1, 28*28).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    l2 = self.loss(outputs, labels)
                    test_loss = test_loss + l2.item()
                    n_samples = n_samples + labels.shape[0]
                    n_correct = n_correct + (outputs.argmax(1) == labels).sum().item()

            avg_train_loss = train_loss / n_total_steps
            avg_test_loss = test_loss / (k+1)
            accuracy = n_correct / n_samples
            print(f'[Completed Epoch {epoch+1}/{self.num_epochs}] Step {i+1}/{n_total_steps} -> Avg Training Loss: {avg_train_loss:.4f}, Avg Test Loss: {avg_test_loss:.4f}, Model Accuracy: {accuracy*100:.2f}%')
            
            #p.plot_epoch(epoch + 1, avg_train_loss, avg_test_loss, accuracy)
            
        input("Press Enter to Finish...")
            
                            
    # plot the train_loss, test_loss and accuracy for each iterations of the generator inputs on the same plot 
    def animate(self):
        plt.clf()
        plt.plot(self.x1, self.y1, label='train_loss')
        plt.plot(self.x1, self.y2, label='test_loss')
        plt.plot(self.x1, self.y3, label='accuracy')
        plt.legend()
        plt.pause(0.001)



        
            






        


            
           
        



