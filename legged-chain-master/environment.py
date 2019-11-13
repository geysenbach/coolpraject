import numpy as np
import math
import functions as f
import parameters
import reward
from scipy.integrate import ode
from scipy.integrate import solve_ivp

def step(time, state, action, cInfo, p):
    """compute a step in the environment

    Arguments:
    time    --  initial time of step
    state   --  state of environment (array of generalized position and velocities [[q],[u]])
    action  --  action applied to step (array of leg and bending torques [[TL],[TB]])
    cInfo   --  information of leg contact state
    p       --  system parameters
    """

    q = state[0:p.dim, [0]]
    u = state[p.dim:, [0]]

    # contact and matrix set-up
    [cc, alpha, tEvent] = f.contacts(time, p)
    cInfo = f.contactinfo(cc, cInfo, alpha, q, p)
    [M, h, MInv, _, Jc, xi] = f.matrixsetup(time, q, u, action, cInfo, p)

    # find constraint-satisfying generalized velocity
    up = f.impact(Jc, MInv, u)
    state[p.dim:, [0]] = up

    # integrate until next leg contact event
    [tSteps, sol] = euler_integration(time, tEvent, state, action, cInfo, p)
    #[tSteps, sol] = ode_integration(time, tEvent, state, action, cInfo, p)
    #[tSteps, sol] = ode_ivp_integration(time, tEvent, state, action, cInfo, p)

    newState = np.transpose(sol[[-1], :])
    rew = reward.val(time, state, action)
    return [tEvent, newState, cInfo, rew, tSteps, sol]

def ode_integration(time, tEvent, state, action, cInfo, p):
    """integrate system of odes using ode"""
    solver = ode(f.dynamics)
    solver.set_integrator('dopri5')
    solver.set_f_params(action, cInfo, p)
    solver.set_initial_value(state, time)
    N = np.floor((tEvent-time)/p.dt).astype('int')
    tSteps = np.linspace(time, tEvent, N)
    sol = np.empty((N, len(state)))
    sol[[0], :] = np.transpose(state)
    k = 1
    while solver.successful() and solver.t < tEvent:
        solver.integrate(tSteps[k])
        sol[[k], :] = np.transpose(solver.y)
        k += 1
    return [tSteps, sol]

def ode_ivp_integration(time, tEvent, state, action, cInfo, p):
    """integrate system of odes using ode_ivp"""
    sol = solve_ivp(lambda t, y: f.dynamics(t,y,action,cInfo,p), [time, tEvent], state[:, 0])
    return [sol.t, np.transpose(sol.y)]

def euler_integration(time, tEvent, state, action, cInfo, p):
    """integrate system of odes using explicit Euler"""
    N = np.floor((tEvent-time)/p.dt).astype('int')
    tSteps = np.linspace(time, tEvent, N)
    sol = np.empty((N, len(state)))
    sol[[0], :] = np.transpose(state)
    k = 1
    while k < N:
        x = np.transpose(sol[[k-1], :])
        dx = f.dynamics(tSteps[k], x, action, cInfo, p)
        sol[[k], :] = np.transpose(x + dx*p.dt)
        k += 1
    return [tSteps, sol]

