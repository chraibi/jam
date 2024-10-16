"""Model definition."""

import numpy as np
import logging
import time
from utils import rk4, euler, heun, init
from dataclasses import dataclass
from typing import Callable, Tuple, IO
from enum import Enum


@dataclass
class SimulationParams:
    """Parameter Dataclass."""

    fps: int = 8  # frames per second
    dt: float = 0.001  # [s] integrator step length
    t_end: int = 3000  # [s] integration time
    N_ped: int = 133  # number of pedestrians
    Length: int = 200  # [m] length of corridor. *Closed boundary conditions*
    v0: float = 1.0  # initial velocity
    av: float = 0.0  # avoidance velocity factor
    a0: float = 1.0  # minimum distance factor
    tau: float = 1.0  # relaxation time


# Once flag
once = True


class SolverType(Enum):
    """Define solver types."""

    RK4 = "RK4"
    EULER = "EULER"
    HEUN = "HEUN"


params = SimulationParams()

# Define the type for the solver
Solver = Callable[[float, float, np.ndarray, Callable], Tuple[float, np.ndarray, int]]


def select_solver(solver_type: SolverType) -> Solver:
    """Select and return the solver function."""
    if solver_type == SolverType.RK4:
        return rk4
    elif solver_type == SolverType.EULER:
        return euler
    elif solver_type == SolverType.HEUN:
        return heun
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


# ------------------------- logging ----------------------------------------


def get_state_vars(
    state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """State variables and dist."""
    x_n = state[0, :]  # x_n
    x_m = np.roll(x_n, -1)  # x_{n+1}
    x_m[-1] += params.Length  # the first one goes ahead by Length
    dx_n = state[1, :]  # dx_n
    dx_m = np.roll(dx_n, -1)  # dx_{n+1}
    dist = x_m - x_n
    if (dist <= 0).any():
        logging.critical("CRITICAL: Negative distances")
    return x_n, x_m, dx_n, dx_m, dist


def model(t: float, state: np.ndarray) -> Tuple[np.ndarray, int]:
    """Calculate the state of the model at time t.

    This function computes the new state of the pedestrians based on their current
    positions and velocities, applying the model's driving and repulsive forces.
    It checks for overlapping agents, returning a flag to indicate whether the
    simulation can continue without issues.

    Args:
        t (float): The current time of the simulation.
        state (np.ndarray): A 2D array where the first row contains the positions
                            of the pedestrians and the second row contains their
                            velocities.

    Returns:
        Tuple[np.ndarray, int]: A tuple where the first element is a 2D array
                                representing the new state of the system (updated
                                positions and velocities), and the second element
                                is a flag indicating the status:
                                - 1 if the update is valid (no overlaps detected),
                                - 0 if overlaps occur (not valid).
    """
    x_n, x_m, dx_n, dx_m, dist = get_state_vars(state)
    if (x_n != np.sort(x_n)).any():  # check if pedestrians are swapping positions
        swapping = x_n != np.sort(x_n)  # swapping is True if there is some swapping
        swapped_dist = x_n[swapping]
        swapped_ped = [i for i, v in enumerate(swapping) if v]
        logging.info(f"Swapped agents: {swapped_ped}")
        logging.info(f"Distances: {swapped_dist}")
        return state, 0

    f_drv = (params.v0 - dx_n) / params.tau
    c = np.e - 1
    Dxn = dist
    Dmin = 2 * params.a0 + params.av * (dx_n + dx_m)
    R = 1 - Dxn / Dmin
    eps = 0.01
    R = eps * np.log(1 + np.exp(R / eps))
    f_rep = -params.v0 / params.tau * np.log(c * R + 1)
    ########################
    a_n = f_rep + f_drv
    #######################
    x_n_new = dx_n
    dx_n_new = a_n
    new_state = np.vstack([x_n_new, dx_n_new])
    return new_state, 1


# ======================================================
def simulation(
    N: int,
    dt: float,
    t_end: float,
    state: np.ndarray,
    once: int,
    solver: Solver,
    file_handler: IO,  # Type hint for a binary file-like object
) -> None:
    """Run the simulation of the model using the specified solver.

    Parameters:
    ----------
    N : int
        The number of pedestrians in the simulation.
    dt : float
        The time step for the simulation.
    t_end : float
        The end time for the simulation.
    state : np.ndarray
        The initial state of the pedestrians (position, velocity, etc.).
    once : int
        A flag indicating whether to run the simulation only once (not used in the current function).
    solver : Solver
        The solver function used for the simulation (e.g., RK4, Euler, Heun).
    file_handler : IO
        The file handler for writing simulation output.
    """
    t = 0
    frame = 0
    iframe = 0
    ids = np.arange(N)
    np.savetxt(file_handler, [], header="id\t time\t x\t v")
    while t <= t_end:
        if frame % (1 / (dt * params.fps)) == 0:
            logging.info(
                f"time={t:6.1f} (<= {t_end:4d}) frame={frame:.3E} v={np.mean(state[1, :]):.6f} "
                f"vmin={min(state[1, :]):.6f} std=+-{np.std(state[1, :]):.6f}"
            )
            T = t * np.ones(N)
            output = np.vstack([ids, T, state]).transpose()
            np.savetxt(file_handler, output, fmt="%d\t %7.3f\t %7.3f\t %7.3f")
            file_handler.flush()

            iframe += 1

        t, state, flag = solver(t, dt, state, model)

        if not flag:
            if once:
                t = t_end - dt
                logging.info(f"t_end {t_end:.2f}")
                once = 0

        frame += 1


# ======================================================
# set up figure and animation

if __name__ == "__main__":
    logfile = "log.txt"
    open(logfile, "w").close()  # touch the file
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    Dyn = float(params.Length) / params.N_ped
    rho = float(params.N_ped) / params.Length
    # ============================
    # ------------------------ Files for data --------------------------------------
    prefix = "%d_av%.2f_v0%.2f" % (params.N_ped, params.av, params.v0)
    filename = "traj_" + prefix + ".txt"

    with open(filename, "wb") as file_handler:
        logging.info("Start initialization with %d peds" % params.N_ped)
        state = init(params.N_ped, params.Length)

        logging.info(
            "Simulation with v0=%.2f, av=%.2f,  dt=%.4f, rho=%.2f"
            % (params.v0, params.av, params.dt, rho)
        )

        logging.info(f"filename: {filename}")
        solver_type = SolverType.EULER
        solver = select_solver(solver_type)
        t1 = time.perf_counter()

        ######################################################
        simulation(
            params.N_ped, params.dt, params.t_end, state, once, solver, file_handler
        )
        ######################################################

        t2 = time.perf_counter()
        logging.info(
            "simulation time %.3f [s] (%.2f [min])" % ((t2 - t1), (t2 - t1) / 60)
        )

        logging.info("close %s" % filename)
