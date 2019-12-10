import numpy as np

a1 = 0.2
a2 = 0.2
v_target = 2.428371031919227
def val(time, state, action, u):
    velocity = np.sqrt(pow(u[0], 2) + pow(u[1], 2))
    return np.maximum((1 - np.abs(v_target - velocity)/a1**(1/a2)), -7500)
