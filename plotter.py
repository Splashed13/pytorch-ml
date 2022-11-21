import matplotlib.pyplot as plt

class Plotter:
    # subplot 1: loss & accuracy vs epoch
    # subplot 2: training loss vs batch

    # change class below to use a subplot with with two methods plot_epoch and plot_batch drawing each subplot respectively
    def __init__(self):
        plt.ion()   
        plt.style.use('seaborn')
        self.x1_epoch = []
        self.y1_epoch = []
        self.y2_epoch = []
        self.y3_epoch = []
        self.x_batch = []
        self.y1_batch = []
        self.y2_batch = []
        self.y3_batch = []
        # init subplot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.fig.suptitle('Model Training Performance')
        plt.show()

    def plot_epoch(self, x, y1, y2, y3):
        self.x1_epoch.append(x)
        self.y1_epoch.append(y1)
        self.y2_epoch.append(y2)
        self.y3_epoch.append(y3)
        self.ax1.cla()
        self.ax1.set_title('Loss & Accuracy vs Epoch')
        self.ax1.set_xlabel('Epoch',)
        self.ax1.set_ylabel('Loss & Accuracy')
        # self.ax1.set_ylim(0, 1.3)
        self.ax2.set_title('Training Loss vs Batch')
        self.ax1.plot(self.x1_epoch, self.y1_epoch, label="Training Loss")
        self.ax1.plot(self.x1_epoch, self.y2_epoch, label="Test Loss")
        self.ax1.plot(self.x1_epoch, self.y3_epoch, label="Test Accuracy", color='orange')
        # set legend in the top right corner
        self.ax1.legend(loc='upper right')
        plt.pause(0.001)

    # input to y can be a single value (loss) or a list of 3 values (loss, test_loss, accuracy)
    def plot_batch(self, x, y):
        self.ax2.cla()
        self.x_batch.append(x)
        self.y1_batch.append(y[0])
        self.y2_batch.append(y[1])
        self.y3_batch.append(y[2])
        self.ax2.set_title('Loss & Accuracy vs Batch Subset')
        self.ax2.set_xlabel('Batch')
        self.ax2.set_ylabel('Loss') 
        self.ax2.plot(self.x_batch, self.y1_batch, label="Training Loss")
        self.ax2.plot(self.x_batch, self.y2_batch, label="Test Loss")
        # made line orange to differentiate from losses
        self.ax2.plot(self.x_batch, self.y3_batch, label="Test Accuracy", color='orange')
        self.ax2.legend(loc='upper right')
        plt.pause(0.001)

if __name__ == '__main__':
    pass
        



    