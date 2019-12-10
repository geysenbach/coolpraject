"""
Centipede Motion environment implemented by Grace Eysenbach and Shravan Nageswaran
Copied from http://incompleteideas.net/sutton/book/code/pole.c
Initial model: Credit to Fabio Giardina - see environment.py reference
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import functions as f
import reward
import scipy.io as sio


class CentipedeEnv(gym.Env):
    """
    Description:
        A centipede has n distinct body segments, each with an angle in between them. With set forces, what is the ideal velocity of each segment that allows the centipede to move far and efficiently.

    Source:
        This environment corresponds to the version of the centipede model defined by the Mahadevan Group at Harvard University.

    Observation:
        The observation state is a 14x1 vector containing q appended to u, or the x and y position, the angular positions of all the body segments, and then the corresponding velocities

    Actions:
        Type: Discrete(2)
        Num	Action
        0	----
        1	---0

        Note: The amount the velocity depends on the angle the of each segment (gained from q).

    Reward:
        The reward chosen is a simple PPO reward which aims to get the resultant velocity as close as possible to a target velocity ~2.48, calculated as described in the paper.

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    """

    def __init__(self):

        self.n = 5  # number of rigid bodies
        self.dim = 2 + self.n  # dimension of generalized coordinates
        self.g0 = 4  # leg length to body length ratio
        self.b0 = 1 / 12  # segment moment of inertia
        self.alpha0 = math.pi / 2  # Initial leg angle
        self.tSpan = np.array([0, 30])  # Episode time span

        g_lbd = 1/1.6  # From Gray, Animal Locomotion p366, "Low gear"
        g_fb = 8.5/1.5  # From Gray, Animal Locomotion p366, "Low gear"
        ph = 1 - 1/(g_lbd*self.n)  # Duty cycle
        dc = 1/(1+g_fb)  # inter-segmental phase difference
        self.ctrlTable = f.sequencer(dc,ph,self)

        # forces and torques
        self.k = 124.3985  # nondimensional stiffness
        self.d = 20.7331  # nondimensional damping
        self.T = -2.448  # nondimensional leg torque

        # compute constant matrices
        self = f.matrixinitializer(self)

        # numerical time step
        self.dt = 2e-3
        self.time = 0

        high = np.array([
            10,
            150,
            np.pi/2,
            np.pi/2,
            np.pi/2,
            np.pi/2,
            np.pi/2,
            3,
            3,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.cInfo = np.array([])

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.store = np.zeros(15, dtype="float")
        self.store = np.array([self.store, self.store])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        time = self.time

        x, y, t1, t2, t3, t4, t5, xd, yd, t1d, t2d, t3d, t4d, t5d = state
        qa = np.array([x, y, t1, t2, t3, t4, t5])
        ua = np.array([xd, yd, t1d, t2d, t3d, t4d, t5d])

        q = qa.reshape((7, 1))
        u = ua.reshape((7, 1))

        # contact and matrix set-up
        [cc, alpha, tEvent] = f.contacts(time, self)
        self.cInfo = f.contactinfo(cc, self.cInfo, alpha, q, self)

        # find constraint-satisfying generalized velocity
        for i in range(2):
            for j in range(6):
                if (np.isnan(self.cInfo[i][j])):
                    self.cInfo[i][j] = 0

        # integrate until next leg contact event
        N = np.floor((tEvent-time)/self.dt).astype('int')
        sol = np.empty((N, len(state)))
        if (N>0):
            tSteps = np.linspace(time, tEvent, N)
            sol[[0], :] = np.transpose(state)
            k = 1
            while k < N:
                x = np.transpose(sol[[k-1], :])
                dx = f.dynamics(tSteps[k], x, action, self.cInfo, self)
                if (np.isnan(dx[0][0])):
                    dx = self.prevdx
                    break
                else:
                    self.prevdx = dx
                sol[[k], :] = np.transpose(x + dx*self.dt)
                k += 1

            newState = np.transpose(sol[[-1], :])
            self.state = newState.reshape((1, 14))[0]

        if (np.isnan(self.state[7])):
            print("Stopping here!")
            print(self.cInfo)
            print(dx)
            print(sol)
            print("continue")

        rew = reward.val(time, state, action, u)
        self.time = self.time + self.dt
        done = self.time >= 100

        if (self.state[7] > 2):
            self.state[7] = 2

        if (self.state[8] > 2):
            self.state[8] = 2

        return self.state, rew, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(14,))
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human'):
        newStore = np.zeros((self.store.shape[0]+1, 15))
        k=0
        for row in self.store:
            newStore[k] = self.store[k]
            k +=1
        self.state[0] = self.state[0]/10000000
        self.state[1] = np.abs(self.state[1]/100)
        newStore[self.store.shape[0]][1:] = self.state
        newStore[self.store.shape[0]][0] = self.time
        self.store = newStore

    def close(self):
        # we can analyze the undulations in MATLAB
        sio.savemat('matlab\sol.mat', {'sol':self.store})
        if self.viewer:
            self.viewer.close()
            self.viewer = None
