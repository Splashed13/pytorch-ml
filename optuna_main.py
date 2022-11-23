import sys
import time
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import dataloader


class Net(nn.Module):
    def __init__(self, l1: int = 120, l2: int = 84):
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


# trainer class for training a model with loss = nn.CrossEntropyLoss() and optimizer = optim.Adam(model.parameters(), lr=learning_rate)
class Trainer:
    def __init__(self, model: nn.Module, loss: torch, optimizer: optim, train_loader: dataloader,
                 test_loader: dataloader, num_epochs: int, device: torch, trial: optuna.trial):
        # fields for plotting the loss and accuracy
        self.trial = trial
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device

    def train(self) -> float:
        print("[Starting Training]")
        # total number of batches
        n_total_steps = len(self.train_loader)
        last_epoch_accuracy = 0
        for epoch in range(self.num_epochs):

            for i, (images, labels) in enumerate(self.train_loader):  # each batch
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                # forward pass
                outputs = self.model(images)
                l = self.loss(outputs, labels)

                # backward pass
                l.backward()  # calculate gradients
                self.optimizer.step()  # update the weight

            # calculate accuracy
            epoch_accuracy = self.test_acc()
            if epoch_accuracy <= last_epoch_accuracy:
                return epoch_accuracy

            last_epoch_accuracy = epoch_accuracy
            print(
                f'[Completed Epoch {epoch + 1}/{self.num_epochs}] Step {i + 1}/{n_total_steps} - Model Accuracy: {epoch_accuracy * 100:.2f}%')
            self.trial.report(epoch_accuracy, epoch)

            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # if epoch = last epoch, return the accuracy
            if epoch == self.num_epochs - 1:
                return epoch_accuracy

    # method that calculates the loss over a random single batch or all batches (if epoch) depending on the input
    def test_acc(self):
        n_correct = 0
        n_samples = 0
        # evaluate at the end of each epoch
        with torch.no_grad():
            for k, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                n_samples = n_samples + labels.shape[0]
                n_correct = n_correct + (outputs.argmax(1) == labels).sum().item()
        avg_accuracy = n_correct / n_samples
        return avg_accuracy


def objective(trial: optuna):
    start_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = {
        # using the rule of thumb 'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'
        # selecting the number of neurons in the hidden layer from a discrete uniform distribution of 15 evenly spaced values between 10 and 400
        # 'l1': trial.suggest_int('l1', 10, 400, step=15),
        # 'l2': trial.suggest_int('l2', 10, 400, step=15),
        # 'num_epochs': trial.suggest_int('num_epochs', 5, 20),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    }

    # define the hyperparameters
    num_epochs = 10
    # batch_size = 25
    # learning_rate = 0.001
    val_split = 0.2

    # define the CIFAR10 dataset
    cifar10_dataset = dataloader.CIFAR10Dataset(params['batch_size'], val_split)

    # define the model
    model = Net().to(device)

    loss = nn.CrossEntropyLoss()  # CrossEntropyLoss combines LogSoftmax and NLLLoss

    # one of the best optimizers for CNNs is Adam
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['learning_rate'])
    # optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    trainer = Trainer(model, loss, optimizer, cifar10_dataset.train_loader, cifar10_dataset.test_loader, num_epochs,
                      device, trial)

    model_accuracy = trainer.train()

    return model_accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.HyperbandPruner(
                                    min_resource=1, max_resource=10, reduction_factor=3
                                ), )
    study.optimize(objective, n_trials=5)
    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)

