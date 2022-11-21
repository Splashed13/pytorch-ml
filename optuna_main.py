import torch
import torch.nn as nn
import torch.optim as optim
from modules import Net
from torch.utils.data import DataLoader
import dataloader
import time
import sys
import optuna
import random

Batch_Size = 25
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
CIFAR10_dataset = dataloader.CIFAR10Dataset(batch_size=Batch_Size, val_split=0.2)

# optuna function to optimize the hyperparameters
def objective(trial):
    # parameter dictionary 
    params = {
        #'hidden_size1': trial.suggest_int('hidden_size1', 10, 100),
        #'hidden_size2': trial.suggest_int('hidden_size2', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    }

    #model = Net(params['hidden_size1'], params['hidden_size2'])
    model = Net()
    accuracy = train_and_evaluate(params, model, CIFAR10_dataset.train_loader, CIFAR10_dataset.test_loader, trial)
    return accuracy


def train_and_evaluate(param:dict, model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, trial:optuna) -> float:
    num_epochs = 5

    # define the loss function and optimizer
    loss = nn.CrossEntropyLoss() # CrossEntropyLoss combines LogSoftmax and NLLLoss

    # adam optimizer 
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate']) #, weight_decay=param['weight_decay'])
    #optimizer = getattr(optim, param['optimizer'])(model.parameters(),lr=param['learning_rate'], weight_decay=param['weight_decay'])

    model = model.to(torch.device("cuda"))
    loss = loss.to(torch.device("cuda"))

    for epoch in range(num_epochs):
        # this will reduce the time for a full epoch by only training on a subset of the data, that subset is 1
        for images, labels in train_loader:
            images = images.to(torch.device("cuda"))
            labels = labels.to(torch.device("cuda"))

            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(images)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
    
        # get accuracy for the test dataset
        n_correct = 0
        n_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(torch.device("cuda"))
                labels = labels.to(torch.device("cuda"))
                outputs = model(images)
                n_samples = n_samples + labels.shape[0]
                n_correct = n_correct + (outputs.argmax(1) == labels).sum().item()

        avg_accuracy = n_correct / n_samples
        print(f"Epoch: {epoch+1}/{num_epochs}, Accuracy: {avg_accuracy*100:.2f}%")
        trial.report(avg_accuracy, epoch)

        if trial.should_prune():

            raise optuna.exceptions.TrialPruned()
        
    return avg_accuracy


if __name__ == '__main__':
    study.optimize(objective, n_trials=100)
    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))