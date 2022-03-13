import math

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from Frame import Frame
from Model import Model


def load_frames(path):
    frames = []
    for fname in glob(path + "/*"):
        frames.append(np.load(fname, allow_pickle=True))
    return frames


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
    relEuclideanMovement = []
    for i in range(1, len(relCoords)):
        prev = relCoords[i - 1]
        cur = relCoords[i]
        relMovement.append((cur[0] - prev[0], cur[1] - prev[1]))
        relEuclideanMovement.append(math.sqrt((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2))

    # print(a1)
    # print(a2)
    # print(relMovement)
    # print(relEuclideanMovement)

    return [a1, a2, relEuclideanMovement]


if __name__ == '__main__':
    X = np.empty((48, 3), dtype=object)
    y = np.zeros(48)
    for i in range(24):
        print(i)
        print(load_data("trajectories/t4"))
        X[i] = load_data("trajectories/t4")
        y[i] = 1
    for i in range(24, 49):
        print(i)
        np.append(X, load_data("trajectories/t4"))
        np.append(y, 1)

    np.save("trainData_X", X, allow_pickle=True)
    np.save("trainData_y", y, allow_pickle=True)

    # model = Model(X, y)
    # model.sequential()
