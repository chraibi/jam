#!/usr/bin/python
import matplotlib
matplotlib.use('macosx')

import matplotlib.pyplot as plt
from sys import argv, exit
import numpy as np

try:
    import pandas as pd
    found_pandas = True
except ImportError:
    found_pandas = False



if len(argv) < 2:
    print("Usage: %s filename" % argv[0])
    exit(-1)
    
ms= 22 #int(argv[2])  #22
mt= 16 #int(argv[3])  #16

lw = 0.15
file1 = argv[1]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
print ("found_pandas: %r" % found_pandas)
print ('loading file %s' % file1)
if found_pandas: # this is about 15 times faster than np.loadtxt()
    A1 = np.array(pd.read_csv(file1, sep="\s+", header=None))
else:
    A1 = np.loadtxt(file1)
    
print ('finished loading')
ids = np.unique(A1[:,0]).astype(int)
frames = A1[:,1]
Length = 200  # warning: this value should be the same as in model.py
figname = file1.split(".txt")[0]

for i in ids[::2]:
    p = A1[ A1[:,0] == i ]
    x1 = p[:,1] #time
    y1 = np.fmod(p[:,2], Length) #x
    abs_d_data = np.abs(np.diff(y1))
    abs_mean = abs_d_data.mean()
    abs_std = abs_d_data.std()
    if abs_std <=0.5*abs_mean:
        T = []
    else:
        T = np.nonzero(abs_d_data > abs_mean + 3*abs_std)[0] 

    start = 0
    for t in T:
        plt.plot(y1[start:t], x1[start:t],'k.',ms=lw, lw=lw, rasterized=True)
        start = t+1

    plt.plot(y1[start:], x1[start:],'k.',ms=lw, lw=lw, rasterized=True)
        
plt.xlabel(r'$x_n$', size=ms)
plt.ylabel(r'$t\; \rm{(s)}$', size=ms)
plt.xticks(fontsize=mt)
plt.yticks(fontsize=mt)
fig.set_tight_layout(True)

#
for e in ["png"]:   #, "png", "eps"]:
    fname = figname + "."+e
    plt.savefig(fname)
    print ('>> figname: %s' % fname)

