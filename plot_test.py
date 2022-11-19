import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob('*.npy')

plt.figure()
for file in files[0:10]:
    dat = np.load(file)
    plt.plot(dat[:,0,0])

plt.show()
