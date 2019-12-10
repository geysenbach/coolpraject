import numpy as np

a1 = 0.2
a2 = 0.2
v_target = 2.428371031919227 #4.671342925015348
epsilon = 10**-8
def val(time, state, action, u):
    """returns a scalar reward"""
    velocity = np.sqrt(pow(u[0], 2) + pow(u[1], 2))
    if (np.abs(v_target - velocity) < epsilon):
        return 0.9
    return (1 - np.abs(v_target - velocity)/a1)**(1/a2)
    #*(np.sign(u[1])) + state[1]*0.1 - np.abs(state[0])
