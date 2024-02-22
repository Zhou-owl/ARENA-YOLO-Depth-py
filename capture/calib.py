import os
from util import *
baseDir = "./capture/"
name_list = os.listdir(baseDir)
name_list = [baseDir+f for f in name_list]
checkboard_size = [5,8]

mtx, dist = computeIntrinsic(name_list, checkboard_size, (8,8))
np.savez("intrinsic_calib.npz", mtx=mtx, dist=dist)

with np.load("./intrinsic_calib.npz") as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

dedistortion(checkboard_size,name_list,  mtx, dist)

