import numpy as np
import parameters
import environment
import functions as f
import scipy.io as sio


p = parameters.callpara()

# initial conditions
q0 = np.zeros((p.dim, 1))  # initial generalized positions
u0 = np.zeros((p.dim, 1))  # initial generalized velocities
state = np.concatenate((q0, u0), axis=0)  # initial state
tStore = np.zeros((1,1))
xStore = np.zeros((1,len(state)))
time = 0

# initial leg ground contact configurations
cInfo = np.array([])
[cc, alpha, tEvent] = f.contacts(0, p)
cInfo = f.contactinfo(cc, cInfo, alpha, q0, p)

# integrate dynamics
for i in range(1, np.size(p.ctrlTable, 0)):
    q = state[0:p.dim, [0]]
    u = state[p.dim:, [0]]

    # define action [[T_leg],[T_bodyBend]]
    action = np.concatenate((p.T*np.ones((p.n,1)), np.zeros((p.n-1,1))),axis=0)

    # do one step in environment
    [time, state, cInfo, reward, tSol, xSol] = environment.step(time, state, action, cInfo, p)
    tSol = np.transpose(np.array([tSol]))
    tStore = np.concatenate((tStore, tSol), axis=0)
    xStore = np.concatenate((xStore, xSol), axis=0)
    print(time)

sol = np.concatenate((tStore, xStore),axis=1)

# save result for animation in matlab
sio.savemat('matlab\sol.mat', {'sol':sol})