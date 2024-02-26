from unittest import TestCase
from control.ilqr import ILQR
import autograd.numpy as np
# import numpy as np


class TestILQR(TestCase):
    def setUp(self):
        self.ilqr = ILQR(dynamics_func=lambda x, u: self.pendulum_dynamics(x, u),
                         running_cost=lambda x, u: self.cost(x, u),
                         terminal_cost=lambda x, u: self.cost(x, u),
                         horizon=20,
                         state_dim=2,
                         input_dim=1)

    def tearDown(self):
        pass

    def test_jacobian_computations(self):
        x_eval_point = np.array([1., 1.])
        u_eval_point = np.array([1.])

        g = 10
        dt = 0.05

        true_f_x = np.array([[1+3/2*g*np.cos(1)*dt**2, dt],
                            [3/2*g*np.cos(1)*dt, 1]])
        self.assertTrue((np.abs(self.ilqr.f_x(x_eval_point, u_eval_point) - true_f_x) < 1e-7).all())

        true_f_u = np.array([[3*dt**2],
                             [3*dt]])
        self.assertTrue((np.abs(self.ilqr.f_u(x_eval_point, u_eval_point) - true_f_u) < 1e-7).all())

        true_f_xx = np.array([
                              [[-3/2*g*np.sin(1)*dt**2, 0],
                               [0, 0]],
                              [[-3/2*g*np.sin(1)*dt, 0],
                               [0, 0]]
                              ])
        self.assertTrue((np.abs(self.ilqr.f_xx(x_eval_point, u_eval_point) - true_f_xx) < 1e-7).all())

        true_f_uu = np.array([[0], [0]])
        self.assertTrue((np.abs(self.ilqr.f_uu(x_eval_point, u_eval_point) - true_f_uu) < 1e-7).all())

        true_c_x = np.array([2*1, 0.2*1])
        self.assertTrue((np.abs(self.ilqr.c_x(x_eval_point, u_eval_point) - true_c_x) < 1e-7).all())

        true_c_u = np.array([0.002*1])
        self.assertTrue((np.abs(self.ilqr.c_u(x_eval_point, u_eval_point) - true_c_u) < 1e-7).all())

        true_c_xx = np.array([[2, 0],
                              [0, 0.2]])
        self.assertTrue((np.abs(self.ilqr.c_xx(x_eval_point, u_eval_point) - true_c_xx) < 1e-7).all())

        true_c_uu = np.array([0.002])
        self.assertTrue((np.abs(self.ilqr.c_uu(x_eval_point, u_eval_point) - true_c_uu) < 1e-7).all())

        # Terminal cost in this case it the same as running cost
        self.assertTrue((np.abs(self.ilqr.ct_x(x_eval_point, u_eval_point) - true_c_x) < 1e-7).all())
        self.assertTrue((np.abs(self.ilqr.ct_xx(x_eval_point, u_eval_point) - true_c_xx) < 1e-7).all())

    def cost(self, x, u):
        costs = self.angle_normalize(x[0]) ** 2 + .1 * x[1] ** 2 + .001 * (u ** 2)
        return costs

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def pendulum_dynamics(x, u):
        """
        Represents dynamics of the system

        th'' = 3g/2*sin(th) + 3u

        in terms of state (th, th').

        :param x: (2-dim array) State of the system (angle of pendulum and angular velocity).
        :param u: (scalar) Control input into the system.
        :return: (2-dim array) The new state.
        """
        th = x[0]
        thdot = x[1]

        g = 10.
        m = 1.
        l = 1.
        dt = 0.05

        u = np.clip(u, -2, 2)[0]

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -8, 8)
        newth = th + newthdot * dt

        x = np.array([newth, newthdot])
        return x
