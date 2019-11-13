import gym
from gym import error, spaces, utils
from gym.utils import seeding

class centipedeEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.state = np.concatenate((q0, u0), axis=0) 
    self.action = np.concatenate((p.T*np.ones((p.n,1)), np.zeros((p.n-1,1))),axis=0)
    self.cInfo = np.array([])
    self.p = parameters.callpara()


  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...

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

