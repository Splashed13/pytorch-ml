import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    # subplot 1: loss & accuracy vs epoch
    # subplot 2: training loss vs batch

    # change class below to use a subplot with with two methods plot_epoch and plot_batch drawing each subplot respectively
    def __init__(self):
        plt.ion()   
        self.x1 = []
        self.y1 = []
        self.y2 = []
        self.y3 = []
        self.xbatch = []
        self.ybatch = []
        # init subplot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.fig.suptitle('Model Training Performance')
        self.ax1.set_title('Loss & Accuracy vs Epoch')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss & Accuracy')
        self.ax1.set_ylim(0, 1)
        self.ax2.set_title('Training Loss vs Batch')
        self.ax2.set_xlabel('Batch')
        self.ax2.set_ylabel('Loss')  
        self.fig.show()

    def plot_epoch(self, x, y1, y2, y3):
        self.x1.append(x)
        self.y1.append(y1)
        self.y2.append(y2)
        self.y3.append(y3)
        self.ax1.plot(self.x1, self.y1, label="train loss")
        self.ax1.plot(self.x1, self.y2, label="train acc")
        self.ax1.plot(self.x1, self.y3, label="val acc")
        self.ax1.legend()
        self.fig.clf()
        plt.pause(0.001)

    def plot_batch(self, x, y):
        self.xbatch.append(x)
        self.ybatch.append(y)
        self.ax2.plot(self.xbatch, self.ybatch, label="train loss")
        self.ax2.legend()
        self.fig.clf()
        plt.pause(0.001)