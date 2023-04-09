import matplotlib.pyplot as plt
import numpy as np
from utils import show_image

param = np.load('./output/lr0.1_hd200_ri0.001/parameters.npz')
w1 = param['w1']
b1 = param['b1']
w2 = param['w2']
b2 = param['b2']

i = 0
for data in w1:
    i += 1
    show_image(data, i)
