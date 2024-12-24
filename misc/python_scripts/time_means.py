import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path


root_dir="../../data/SVMNEW/RBF/test_0/time_info"


total=np.ndarray(7)
cnt=0
for foldername,subfolders,filenames in os.walk(root_dir):
    print(foldername)
    print(subfolders)
    print(filenames)
    for file in filenames:
        print(foldername+"/"+file)
        data=np.genfromtxt(foldername+"/"+file+"",delimiter=",",dtype="str")
        print(data)
        data=data[:,1].astype(float)
        print(data)
        total+=data
        cnt=cnt+1

total=total/cnt
print(total)
print(total[0]+total[1]+total[2]+total[3]+total[4])
print(total[5])
print(total[6])



