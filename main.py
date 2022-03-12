import numpy as np
import pickle
from tenserflow import keras


def loader():
    a = np.load("trajectories/t1/trajectory1_frame0000.npy", allow_pickle=True)
    print(a)


if __name__ == '__main__':
    loader()

