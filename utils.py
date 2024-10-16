"""Utility functions for solvers and initialization."""

import numpy as np
from typing import Callable, Tuple


def init(N: int, Length: float) -> np.ndarray:
    """
    Initialize positions and velocities of N agents along a 1D space of given Length.

    Args:
        N (int): The number of agents.
        Length (float): The length of the 1D space.

    Returns:
        np.ndarray: A 2D array with positions in the first row and velocities in the second row.
    """
    error = 0.1  # Error applied to the first agent's position
    eps_x = 0.0  # Small perturbation for positions (set to != 0 if needed)
    eps_v = 0.0  # Small perturbation for velocities (set to != 0 if needed)

    shift = float(Length) / N  # Distance between agents
    x_n = 0.5 * shift + shift * np.arange(N) + np.random.uniform(0, eps_x, N)
    dx_n = np.random.uniform(0, eps_v, N)  # Small random velocities

    # Adjust the first position to include error
    x_n[0] += error

    return np.vstack([x_n, dx_n])


def rk4(
    x: float,
    h: float,
    y: np.ndarray,
    f: Callable[[float, np.ndarray], Tuple[np.ndarray, int]],
) -> Tuple[float, np.ndarray, int]:
    """
    Fourth-order Runge-Kutta solver for differential equations.

    Args:
        x (float): The current x (time) value.
        h (float): The step size for the integration.
        y (np.ndarray): The current state of the system.
        f (Callable): The function that computes the derivative of y at x.

    Returns:
        Tuple[float, np.ndarray, int]: Updated x value, updated y value, and a flag (1 if successful).
    """
    k1 = h * f(x, y)[0]
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)[0]
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)[0]
    k4 = h * f(x + h, y + k3)[0]

    return x + h, y + (k1 + 2 * (k2 + k3) + k4) / 6.0, 1


# ======================================================
def euler(
    x: float,
    h: float,
    y: np.ndarray,
    f: Callable[[float, np.ndarray], Tuple[np.ndarray, int]],
) -> Tuple[float, np.ndarray, int]:
    """
    Euler method for solving ordinary differential equations.

    Args:
        x (float): The current x (time) value.
        h (float): The step size for the integration.
        y (np.ndarray): The current state of the system.
        f (Callable): The function that computes the derivative of y at x.

    Returns:
        Tuple[float, np.ndarray, int]: Updated x value, updated y value, and a flag indicating success.
    """
    y_new, flag = f(x, y)
    return x + h, y + h * y_new, flag


def heun(
    x: float,
    h: float,
    y: np.ndarray,
    f: Callable[[float, np.ndarray], Tuple[np.ndarray, int]],
) -> Tuple[float, np.ndarray, int]:
    """
    Heun's method (Improved Euler) for solving ordinary differential equations.

    Args:
        x (float): The current x (time) value.
        h (float): The step size for the integration.
        y (np.ndarray): The current state of the system.
        f (Callable): The function that computes the derivative of y at x.

    Returns:
        Tuple[float, np.ndarray, int]: Updated x value, updated y value, and a flag indicating success.
    """
    k1, flag1 = f(x, y)
    k2, flag2 = f(x + h, y + h * k1)

    return x + h, y + h / 2.0 * (k1 + k2), flag1 * flag2
