import numpy as np
import matplotlib.pyplot as plt
import glob


root="../../data/SVMNEW/RBF/test_"


exp_list={
    "0":"0.001",
    "1":"0.005",
    "2":"0.01",
    "3":"0.025",
    "4":"0.05",
    "5":"0.075",
    "6":"0.1",
    "7":"0.5",
    "8":"1",
    "9":"5"
}

exp_list={
    "0":"0.1",
    "1":"0.2",
    "2":"0.3",
    "3":"0.4",
    "4":"0.5",
}

markers=['o','s','D','X','v','^','<','>','P','*']

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
id=0;
for idx,label in exp_list.items():
    template=""
    name=root+idx+"/log.csv"
    data=np.loadtxt(name,delimiter=",",skiprows=1)
    # Accuracy
    axs[0,0].plot(data[:,0],data[:,1],marker=markers[id],label="sigma:"+label)
    axs[0,1].plot(data[:,0],data[:,3],marker=markers[id],label="sigma:"+label)
    # Hinge
    axs[1,0].plot(data[:,0],data[:,2],marker=markers[id],label="sigma:"+label)
    axs[1,1].plot(data[:,0],data[:,4],marker=markers[id],label="sigma:"+label)
    id+=1

fig.suptitle("1v1 RBF Kernel")
axs[0,0].legend()
axs[0,0].set_xlabel("C")
axs[0,0].set_xscale('log')
axs[0,0].set_ylabel("Training set Accuracy")

axs[0,1].legend()
axs[0,1].set_xlabel("C")
axs[0,1].set_xscale('log')
axs[0,1].set_ylabel("Test set Accuracy")

axs[1,0].legend()
axs[1,0].set_xlabel("C")
axs[1,0].set_xscale('log')
axs[1,0].set_ylabel("Training set Hinge Loss")

axs[1,1].legend()
axs[1,1].set_xlabel("C")
axs[1,1].set_xscale('log')
axs[1,1].set_ylabel("Test set Hinge Loss")


fig.tight_layout()
plt.show()






