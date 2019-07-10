import cv2
from matplotlib import pyplot as plt
import os


class ArtworkClassifier:

    def __init__(self, imageinput):
        self.image = imageinput


    @classmethod
    def fromfile(cls, path, name):
        return cls(cv2.imread(os.path.join(path,name)))


    def isartwork(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()
        return True

