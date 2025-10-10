import os
import numpy as np
import h5py

path = "/mnt/c/Users/onion/Documents/data/m_train.mat"

f = h5py.File(path, "r")
print(list(f.keys()))
