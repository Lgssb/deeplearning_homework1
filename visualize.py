import matplotlib.pyplot as plt
import numpy as np

# 参数可视化

param = np.load('./output/lr0.1_hd200_ri0.001/parameters.npz')
w1 = param['w1']
w2 = param['w2']

plt.imshow(w1, cmap='RdBu')
plt.clim(-0.05, 0.05)
plt.savefig(f'./Parameters_images/w1.jpg')

plt.imshow(w2, cmap='RdBu')
plt.clim(-0.5, 0.5)
plt.savefig(f'./Parameters_images/w2.jpg')


