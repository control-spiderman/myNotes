import numpy as np


try:
    import cPickle
except BaseException:
    import _pickle as cPickle

data = cPickle.load(open("goal_set.p",'rb+'),encoding='utf-8')
print(data)
np.save("fileName.npy",data)

# from scipy import io


data1 = np.load("fileName.npy",allow_pickle=True)
import pprint

import json

data1 = data1.reshape(1)
print(len(data1))
np.save("dataset.txt",data1)
