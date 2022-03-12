import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from Frame import Frame


def load(path):
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


if __name__ == '__main__':
    frames = load("trajectories/t4")
    # plot(frames)
    a1 = a2 = []
    relCoords = []
    for f in frames:
        frame = Frame(f)
        areas = frame.get_areas()
        if len(areas) > 2:
            print("more than 2 blobs detected")
        elif len(areas) == 2:
            # update area info
            a1.append(areas[0])
            a2.append(areas[1])

            # update relative coordinate info
            coords = frame.get_centers()
            x = coords[0][0] - coords[1][0]
            y = coords[0][1] - coords[1][1]
            relCoords.append((x, y))

        # frame.show()
        # frame.show_contour()
    # take area to be average of larges 20 frames

    a1 = a1.sort()
    a2 = a2.sort()
    print(relCoords)
