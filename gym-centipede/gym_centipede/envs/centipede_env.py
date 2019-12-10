import math
import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np
import gym_centipede.envs.functions as f
import gym_centipede.envs.parameters
import math
import gym_centipede.envs.reward
from scipy.integrate import ode
from scipy.integrate import solve_ivp

class CentipedeEnv(gym.Env):

    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second' : 50
    # }

    def __init__(self):

        self.n = 3  # number of rigid bodies
        self.dim = 2 + self.n  # dimension of generalized coordinates
        self.g0 = 4  # leg length to body length ratio
        self.b0 = 1 / 12  # segment moment of inertia
        self.alpha0 = math.pi / 2  # Initial leg angle
        self.tSpan = np.array([0, 30])  # Episode time span

        # g_lbd = 1/1.6  # From Gray, Animal Locomotion p366, "Low gear"
        # g_fb = 8.5/1.5  # From Gray, Animal Locomotion p366, "Low gear"

        self.ctrlTable = f.sequencer(1/(1+(8.5/1.5)),1 - 1/((1/1.6)*3),self)

        # forces and torques
        self.k = 124.3985  # nondimensional stiffness
        self.d = 20.7331  # nondimensional damping
        self.T = -2.448  # nondimensional leg torque

        # compute constant matrices
        # p = f.matrixinitializer(p)

        # numerical time step
        self.dt = 2e-3
        self.time = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            10,
            150,
            np.pi/2,
            np.pi/2,
            np.pi/2,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        """compute a step in the environment

        Arguments:
        time    --  initial time of step
        state   --  state of environment (array of generalized position and velocities [[q],[u]])
        action  --  action applied to step (array of leg and bending torques [[TL],[TB]])
        cInfo   --  information of leg contact state
        p       --  system parameters
        """
        state = self.state
        time = self.time

        x, y, t1, t2, t3, x_dot, y_dot, t1_dot, t2_dot, t3_dot = state
        q = np.array([x, y, t1, t2, t3])
        u = np.array([x_dot, y_dot, t1_dot, t2_dot, t3_dot])

        cInfo = np.zeros((10, 10))

        class para:
            pass

        p = para()

        # parameters
        p.n = self.n  # number of rigid bodies
        p.dim = 2 + p.n  # dimension of generalized coordinates
        p.g0 = self.g0  # leg length to body length ratio
        p.b0 = self.b0  # segment moment of inertia
        p.alpha0 = self.alpha0  # Initial leg angle
        p.tSpan = self.tSpan  # Episode time span

        p.ctrlTable = self.ctrlTable

        # forces and torques
        p.k = self.k  # nondimensional stiffness
        p.d = self.d  # nondimensional damping
        p.T = self.T  # nondimensional leg torque

        # compute constant matrices
        p = f.matrixinitializer(p)

        # numerical time step
        p.dt = self.dt

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
        self.time = self.time + 0.00001

        return [tEvent, newState, cInfo, rew, tSteps, sol]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(10,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
