import os
# from numpy import *
import numpy as np

def sh(script):
    # run bash command
    os.system("bash -c '%s'" % script)

#======================================================
def init(N, Length):
    error = 0.1 #0.0001 # =1cm
    eps_x = 0  # small perturbation. Set to !=0 if necessary
    eps_v = 0  # small perturbation. Set to !=0 if necessary
    shift = float(Length)/N
    x_n = 0.5*shift + shift*np.arange(N) + np.random.uniform(0, eps_x, N)
    dx_n = np.random.uniform(0, eps_v, N) #zeros(N)
    x_n[0] += error

    #x_n[0]=0
    #x_n[-1]=Length-3
    return np.vstack([x_n, dx_n])
#======================================================
def rk4(x, h, y, f):
    k1 = h * f(x, y)[0]
    k2 = h * f(x + 0.5*h, y + 0.5*k1)[0]
    k3 = h * f(x + 0.5*h, y + 0.5*k2)[0]
    k4 = h * f(x + h, y + k3)[0]
    return x + h, y + (k1 + 2*(k2 + k3) + k4)/6.0, 1
#======================================================
def euler(x, h, y, f):
    y_new, flag = f(x, y)
    return x + h, y + h * y_new, flag
#======================================================
def heun(x, h, y, f):
    k1, flag1 = f(x, y)
    k2, flag2 = f(x + h, y + h*k1)
    return x + h, y + h/2.0*(k1 + k2), flag1*flag2
#======================================================
#======================================================


