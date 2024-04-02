import numpy as np
import threading
import math

class Ant:
    """
    Ant's will hold the data frames of the data set as their genetic.
    """

    def __init__(self, action, index):
        # saves the label i.e. the colony == cluster to which this ant belongs
        self.label = -1

        # the template of this Ant
        self.template = np.random.normal(0.5, 0.2)

        # age of the ant, updated during a change in acceptance threshold
        self.age = 0

        # store the index of the genetic
        # i.e. index of this data frame inside the data set and inside AntClut
        self.index = index 

        # m is an estimator which is reflecting how successful this ant is
        self.m = 0

        self.label = self.define_label(action)
        # locker for multithreading
        self.lock = threading.Lock()

    def define_label(self, data):
        return 1 if data >= self.template else 0

    # -----------------
    #       getter
    # -----------------
    def get_label(self):
        self.lock.acquire()
        value = self.label
        self.lock.release()
        return value

    def get_m(self):
        self.lock.acquire()
        value = self.m
        self.lock.release()
        return value
    
    def get_age(self):
        self.lock.acquire()
        value = self.age
        self.lock.release()
        return value

    # -----------------
    #      setter
    # -----------------
    def set_label(self, label):
        self.lock.acquire()
        self.label = label
        self.lock.release()

    def set_m(self, m):
        self.lock.acquire()
        self.m = m
        self.lock.release()


    def set_age(self, age):
        self.lock.acquire()
        self.age = age
        self.lock.release()