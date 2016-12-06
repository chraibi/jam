#!/usr/bin/python
#-------------------------------------------------------------
# from numpy import *
import numpy as np
# import os
import logging
import time
import sys
from utils import *
from shutil import copy
#---------------------- Parameter ---------------------------------------
fps = 8        # frames per second
dt = 0.001     # [s] integrator step length
t_end = 3000     # [s] integration time
N_ped = 133    # number of pedestrians delta YN= 1.5
Length = 200   # [m] length of corridor. *Closed boundary conditions*
#========================= SOLVER
RK4 = 0        # 1 --> RK4.
EULER = 1      # if RK4==0 and EULER==0 ---> Heun
HEUN = 0
once = 1
#------------------------- logging ----------------------------------------
rho = float(N_ped)/Length
#======================================================
def get_state_vars(state):
    """
    state variables and dist
    """
    x_n = state[0, :]       # x_n
    x_m = np.roll(x_n, -1)    # x_{n+1}
    x_m[-1] += Length    # the first one goes ahead by Length
    dx_n = state[1, :]      # dx_n
    dx_m = np.roll(dx_n, -1)  # dx_{n+1}
    dist = x_m - x_n
    if (dist <= 0).any():
        logging.critical("CRITICAL: Negative distances")
    return x_n, x_m, dx_n, dx_m, dist
#======================================================
def model(t, state):
    """
    log model
    """
    x_n, x_m, dx_n, dx_m, dist = get_state_vars(state)
    if  (x_n != np.sort(x_n)).any(): # check if pedestrians are swaping positions
        swapping = (x_n != np.sort(x_n)) # swapping is True if there is some swapping
        swaped_dist = x_n[swapping]
        swaped_ped = [i for i, v in enumerate(swapping) if v == True]

        print ('swaped_peds' % swaped_ped)
        print ('distances '% swaped_dist)
        return state, 0

    f_drv = (v0 - dx_n)/tau
    c = np.e - 1
    Dxn = dist 
    Dmin = 2*a0 + av*(dx_n+dx_m)
    R = 1 - Dxn/Dmin
    eps = 0.01
    R = eps*np.log(1+np.exp(R/eps))
    f_rep = -v0/tau * np.log(c*R+1)
    ########################
    a_n = f_rep + f_drv
    #######################
    x_n_new = dx_n
    dx_n_new = a_n
    new_state = np.vstack([x_n_new, dx_n_new])
    return new_state, 1
#======================================================
def simulation(N, dt, t_end, state, once, Length, f):
    global positions, velocities
    t = 0
    frame = 0
    iframe = 0
    ids = np.arange(N_ped)
    while t <= t_end:
        if frame%(1/(dt*fps)) == 0: # one frame per second
            logging.info("time=%6.1f (<= %4d) frame=%3.3E v=%f  vmin=%f  std=+-%f"%(
                t, t_end, frame, np.mean(state[1, :]), min(state[1, :]), np.std(state[1, :])))
            T = t*np.ones(N_ped)
            output = np.vstack([ids, T, state]).transpose()
            np.savetxt(f, output, fmt="%d\t %f\t %f\t %f")  # not python3 compatible
            f.flush()

            iframe += 1

        if RK4: # Runge-Kutta
            t, state, flag = rk4(t, dt, state, model)
        elif EULER: # Euler
            t, state, flag = euler(t, dt, state, model)
        elif HEUN: #  Heun
            t, state, flag = heun(t, dt, state, model)

        if not flag:
            if once:
                t = t_end - dt
                print ('t_end %f' % t_end)
                if RK4:
                    logging.info("Solver:\t RK4")
                elif EULER: # Euler
                    logging.info("Solver:\t EULER")
                else:
                    logging.info("Solver:\t HEUN")
                once = 0

        frame += 1
#======================================================
# set up figure and animation

if __name__ == "__main__":
    logfile = "log.txt"
    open(logfile, "w").close() # touch the file
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s')

    Dyn = float(Length)/N_ped
    #============================
    av = 0.0
    v0 = 1.0
    a0 = 1.0
    tau = 1.0
    #============================
    #------------------------ Files for data --------------------------------------
    prefix = "%d_av%.2f_v0%.2f"%(N_ped, av, v0)
    filename = "traj_" + prefix + ".txt"
    f = open(filename, 'w+')
    

    
#    write_geometry()
    logging.info("start initialisation with %d peds"%N_ped)
    state = init(N_ped, Length)
    positions = np.copy(state[0])
    velocities = np.copy(state[1])


                   
    logging.info("simulation with v0=%.2f, av=%.2f,  dt=%.4f, rho=%.2f"%(v0, av, dt, rho))

    print ('filename %s' % filename)
    t1 = time.clock()
    ######################################################
    simulation(N_ped, dt, t_end, state, once, Length, f)
    ######################################################
    # a = anim.animate_solution(u, peds, targets)
    t2 = time.clock()
    logging.info("simulation time %.3f [s] (%.2f [min])"%((t2-t1), (t2-t1)/60))
    
    logging.info("close %s"%filename)
    f.close()
    
    
    
