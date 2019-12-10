import numpy as np

def val(time, state, action, u):
    """returns a scalar reward"""
    return np.sqrt(pow(u[0], 2) + pow(u[1], 2))*(np.sign(u[1])) + state[1]*0.1 - np.abs(state[0])
