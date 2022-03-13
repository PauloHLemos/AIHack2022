import math

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from Frame import Frame
from Model import Model


# animates motion of given directory of frames
def plot(frames):
    fig = plt.figure()

    viewer = fig.add_subplot(111)
    fig.show()
    # turn on interactive mode
    plt.ion()

    for i in range(0, max(len(frames), 100), 10):
        viewer.clear()
        viewer.imshow(frames[i], interpolation='nearest')
        # plt.imshow(frame, interpolation='nearest')
        plt.pause(.05)
        fig.canvas.draw()


def load_frames(path):
    frames = []
    for fname in glob(path + "/*"):
        frames.append(np.load(fname, allow_pickle=True))
    return frames


def load_data(dir):
    frames = load_frames(dir)
    # plot(frames)
    areas1 = []
    areas2 = []
    relCoords = []
    for f in frames:
        frame = Frame(f)
        areas = frame.get_areas()
        if len(areas) > 2:
            print("more than 2 blobs detected")
        elif len(areas) == 2:
            # update area info
            areas1.append(areas[0])
            areas2.append(areas[1])

            # update relative coordinate info
            coords = frame.get_centers()
            x = coords[0][0] - coords[1][0]
            y = coords[0][1] - coords[1][1]
            relCoords.append((x, y))

        # frame.show()
        # frame.show_contour()

    # take area to be average of larges 20 frames
    areas1.sort()
    areas1 = areas1[-20:]
    areas2.sort()
    areas2 = areas2[-20:]
    a1 = sum(areas1) / len(areas1)
    a2 = sum(areas2) / len(areas2)

    # get relative movement from relative coordinates
    relMovement = []
    relEuclideanMovement = [0] * len(relCoords)
    for i in range(1, len(relCoords)):
        prev = relCoords[i - 1]
        cur = relCoords[i]
        relMovement.append((cur[0] - prev[0], cur[1] - prev[1]))
        relEuclideanMovement[i] = (math.sqrt((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2))

    # print(a1)
    # print(a2)
    # print(relMovement)
    # print(relEuclideanMovement)

    return [a1, a2] + relEuclideanMovement[:100]


def load():
    X = np.zeros((49, 102))
    y = np.zeros(49)
    for i in range(1, 24):
        print(i)
        x_temp = load_data("trajectories/t" + str(i))
        if len(x_temp) == 102:
            print(x_temp)
            X[i] = x_temp
            y[i] = 1
    for i in range(24, 49):
        print(i)
        x_temp = load_data("trajectories/t" + str(i))
        if len(x_temp) == 102:
            print(x_temp)
            X[i] = x_temp

    np.save("trainData_X", X, allow_pickle=True)
    np.save("trainData_y", y, allow_pickle=True)


if __name__ == '__main__':
    # frames = load_frames("trajectories/t1")
    # plot(frames)
    # load()
    X = np.load("trainData_X.npy", allow_pickle=True)
    y = np.load("trainData_y.npy", allow_pickle=True)

    X_train = np.concatenate([X[:15], X[23:39]])
    X_test  = np.concatenate([X[15:23], X[39:48]])

    y_train = np.append(y[:15], y[23:39])
    y_test  = np.append(y[15:23], y[39:48])

    # remove zero rows
    zeroRows = np.all(X_train == 0, axis=1)
    X_train = X_train[~zeroRows]
    y_train = y_train[~zeroRows]

    zeroRows = np.all(X_test == 0, axis=1)
    X_test = X_test[~zeroRows]
    y_test = y_test[~zeroRows]
    #
    # print(X_test)
    # print(y_test)

    model = Model(X, y, X_test, y_test)
    model.sequential()
