import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self):
        self.x1 = []
        self.y1 = []
        self.y2 = []
        self.y3 = []
        plt.ion()     
        plt.ylim(0, 1)
        plt.show()

        