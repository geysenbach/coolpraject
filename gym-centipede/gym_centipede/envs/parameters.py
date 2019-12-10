import numpy as np
import math
import gym_centipede.envs.functions as f

def callpara():
    """Define system parameters"""
    class para:
        pass

    p = para()

    # parameters
    p.n = 21  # number of rigid bodies
    p.dim = 2 + p.n  # dimension of generalized coordinates
    p.g0 = 1.5  # leg length to body length ratio
    p.b0 = 1 / 12  # segment moment of inertia
    p.alpha0 = math.pi / 2  # Initial leg angle
    p.tSpan = np.array([0, 30])  # Episode time span

    g_lbd = 1/1.6  # From Gray, Animal Locomotion p366, "Low gear"
    g_fb = 8.5/1.5  # From Gray, Animal Locomotion p366, "Low gear"
    ph = 1 - 1/(g_lbd*p.n)  # Duty cycle
    dc = 1/(1+g_fb)  # inter-segmental phase difference
    p.ctrlTable = f.sequencer(dc,ph,p)

    # forces and torques
    p.k = 124.3985  # nondimensional stiffness
    p.d = 20.7331  # nondimensional damping
    p.T = -2.488  # nondimensional leg torque

    # compute constant matrices
    p = f.matrixinitializer(p)

    # numerical time step
    p.dt = 2e-3

    return p
