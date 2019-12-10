import numpy as np
import parameters
import environment
import functions as f
import scipy.io as sio
import matplotlib.pyplot as plt


p = parameters.callpara()

# initial conditions
q0 = np.zeros((p.dim, 1))  # initial generalized positions
u0 = np.zeros((p.dim, 1))  # initial generalized velocities
state = np.concatenate((q0, u0), axis=0)  # initial state
# print(len(q0))
tStore = np.zeros((1,1))
xStore = np.zeros((1,len(state)))
rStore = np.zeros((1,1))
time = 0

# initial leg ground contact configurations
cInfo = np.array([])
[cc, alpha, tEvent] = f.contacts(0, p)
cInfo = f.contactinfo(cc, cInfo, alpha, q0, p)
velocities = np.zeros(2500)#np.size(p.ctrlTable, 0)-1)
appv = np.zeros(2500) #np.size(p.ctrlTable, 0)-1)
rewards = np.zeros(2500) #np.size(p.ctrlTable, 0)-1)
power = np.zeros(2500)#np.size(p.ctrlTable, 0)-1)
cots = np.zeros(2500)#np.size(p.ctrlTable, 0)-1)
i_arr = range(2500)#np.size(p.ctrlTable, 0)-1)

# integrate dynamics
for i in range(1, 2500): #np.size(p.ctrlTable, 0)):
    q = state[0:p.dim, [0]]
    u = state[p.dim:, [0]]

    # define action [[T_leg],[T_bodyBend]]
    action = np.concatenate((p.T*np.ones((p.n,1)), np.zeros((p.n-1,1))),axis=0)
    actions = np.array([])
    max_reward = -100000
    max_action = action
    for action in actions:
        [_, _, _, reward, _, _] = environment.step(time, state, action, cInfo, p)
        if reward > max_reward:
            max_reward = reward
            max_action = action

    # do one step in environment
    [time, state, cInfo, reward, tSol, xSol] = environment.step(time, state, max_action, cInfo, p)
    # rewards[i-1] = reward
    # print(time, reward)
    power[i-1] = np.abs(p.T)*(np.sum(np.abs(u[2:])))
    cots[i-1] = power[i-1]/(np.sqrt(pow(u[0], 2) + pow(u[1], 2)))
    velocities[i-1] = (np.sqrt(pow(u[0], 2) + pow(u[1], 2)))
    appv[i-1] = power[i-1]/velocities[i-1]
    tSol = np.transpose(np.array([tSol]))
    reward = np.transpose(np.array([reward]))
    tStore = np.concatenate((tStore, tSol), axis=0)
    xStore = np.concatenate((xStore, xSol), axis=0)
    rStore = np.concatenate((rStore, reward), axis=0)
    print(time)
    # print(state)

# print(rStore)
# rStore /= tStore
avg_reward = np.average(rStore) - q[1]/100
# print(avg_reward)
sol = np.concatenate((tStore, xStore),axis=1)

average_power = np.average(power)
avg_powerline = np.array([average_power for j in i_arr])

average_cot = np.average(cots)
avg_cotline = np.array([average_cot for j in i_arr])

# average_appv = np.average(appvs)
# avg_appvline = np.array([average_appv for j in i_arr])

average_velocity = np.average(velocities)
# print("avg")
# print(average_velocity)
avg_velocityline = np.array([average_velocity for j in i_arr])

plt.figure()
# plt.plot(i_arr, rewards)
plt.plot(i_arr, power)
plt.plot(i_arr, avg_powerline, label="average_power")
plt.title("Power Efficiency of Centipede over Time")
plt.show()

plt.figure()
# plt.plot(i_arr, rewards)
plt.plot(i_arr, cots)
plt.plot(i_arr, avg_cotline, label="average_COT")
plt.title("Cost of Transport of Centipede over Time")
plt.show()

# plt.figure()
# # plt.plot(i_arr, rewards)
# plt.plot(i_arr, appvs)
# plt.plot(i_arr, avg_cotline, label="average_COT")
# plt.title("Cost of Transport of Centipede over Time")
# plt.show()

plt.figure()
plt.plot(i_arr, velocities)
plt.plot(i_arr, avg_velocityline, label="average_velocity")
plt.title("Velocity of Centipede over Time")
plt.show()

# print("Average")
# print(np.average(reward))

# save result for animation in matlab
sio.savemat('matlab\sol.mat', {'sol':sol})
