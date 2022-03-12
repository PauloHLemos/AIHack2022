import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Frame:
    def __init__(self, frame):
        self.frame = frame
        self.height, self.width = frame.shape

        # convert frame to cv greyscale picture
        im = np.array(frame * 255, dtype=np.uint8)
        self.gray = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
        self.out = self.__get_contour()

    def __get_contour(self):
        _, threshold = cv.threshold(self.gray, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # filter out noise
        filtered = list(filter(lambda c: cv.contourArea(c) >= 1000, contours))
        self.contours = filtered

        # initiate empty image, and draw contour
        out = np.full((self.height, self.width), 0, dtype=np.uint8)
        cv.drawContours(out, filtered, -1, 255, 1)

        return out

    def get_contours_info(self):
        centers = []
        for c in self.contours:
            M = cv.moments(c)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append(((cX, cY), cv.contourArea(c)))
        return centers

    def get_centers(self):
        centers = []
        for c in self.contours:
            M = cv.moments(c)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        return centers

    def get_areas(self):
        areas = []
        for c in self.contours:
            areas.append(cv.contourArea(c))
        return areas

    def show(self):
        plt.plot(self.gray)
        cv.imshow('image', self.gray)
        c = cv.waitKey()

    def show_contour(self):
        plt.plot(self.out)
        cv.imshow('image', self.out)
        c = cv.waitKey()

    # out = np.full((frame.height, frame.width), 0, dtype=np.uint8)
    # cv.circle(out, (cX, cY), 5, (255, 255, 255), -1)
    # cv.putText(out, "centroid", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #
    # cv.imshow("Image", out)
    # cv.waitKey(0)