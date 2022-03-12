import numpy as np
import matplotlib.pyplot as plt
from glob import glob


class Blob:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius