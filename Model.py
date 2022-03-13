import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.models import Sequential
from keras.layers import Dense


class Model():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sequential(self):
        # define the keras model
        model = Sequential()
        model.add(Dense(12, input_dim=102, activation='relu'))
        model.add(Dense(102, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(self.X, self.y, epochs=150, batch_size=10)
        # evaluate the keras model
        _, accuracy = model.evaluate(self.X, self.y)
        print('Accuracy: %.2f' % (accuracy*100))