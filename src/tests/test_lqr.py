from unittest import TestCase
from control.lqr import InfiniteHorizonDiscreteLQR, FiniteHorizonDiscreteLQR
import numpy as np


class TestInfiniteHorizonLQR(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_idare_solution(self):
        """
        Ensure that the correct solution to the discrete-time algebraic Riccati equation (DARE) is computed.
        Example is taken from the documentation of the "idare" function in MatLab:
        https://www.mathworks.com/help/control/ref/idare.html#mw_5ae2107f-c2c0-457d-b8ed-ff9eb122a73c.
        """
        self.A = np.array([[-0.9, -0.3], [0.7, 0.1]])
        self.B = np.array([[1], [1]])
        self.Q = np.array([[1, 0], [0, 3]])
        self.R = 0.1

        self.state_dim = 2
        self.input_dim = 1

        self.lqr = InfiniteHorizonDiscreteLQR(A=self.A,
                                              B=self.B,
                                              Q=self.Q,
                                              R=self.R)

        true_P = np.array([[4.7687, 0.9438], [0.9438, 3.2369]])
        print("Computed P: \n", self.lqr.P)
        print("True P: \n", true_P)
        self.assertTrue((np.abs(self.lqr.P - true_P) < 1e-3).all())

    def test_analytical_dare_solution(self):
        """
        Ensure that the correct solution to the discrete-time algebraic Riccati equation (DARE) is computed.
        The solution can easily be computed by hand
        """
        self.A = np.array([[0, 0], [1, 0]])
        self.B = np.array([[0], [1]])
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = 1

        self.state_dim = 2
        self.input_dim = 1

        self.lqr = InfiniteHorizonDiscreteLQR(A=self.A,
                                              B=self.B,
                                              Q=self.Q,
                                              R=self.R)

        true_P = np.array([[3/2, 0], [0, 1]])
        print("Computed P: \n", self.lqr.P)
        print("True P: \n", true_P)
        self.assertTrue((np.abs(self.lqr.P - true_P) < 1e-7).all())

    def test_simulate_trajectory(self):
        """
        Ensure that the trajectory of a discrete-time linear system under the optimal control sequence converges.
        """
        self.A = np.array([[1.01, 0.01, 0],
                           [0.01, 1.01, 0.01],
                           [0, 0.01, 1.01]])
        self.B = np.identity(3)
        self.Q = 1e-3 * np.identity(3)
        self.R = np.identity(3)
        self.horizon = 1100

        self.state_dim = 3
        self.input_dim = 3

        self.lqr = InfiniteHorizonDiscreteLQR(A=self.A,
                                              B=self.B,
                                              Q=self.Q,
                                              R=self.R)

        J, diffs = self.lqr.simulate_trajectory(horizon=self.horizon)
        for i in range(1000, 1100):
            self.assertTrue(np.abs(diffs[i]) < 1e-9)

    def test_dare_override(self):
        """
        Ensure that when overriding A, B in compute_dare_solution(), A and B stay the same.
        """
        self.A = np.array([[1.01, 0.01, 0],
                           [0.01, 1.01, 0.01],
                           [0, 0.01, 1.01]])
        self.B = np.identity(3)
        self.Q = 1e-3 * np.identity(3)
        self.R = np.identity(3)
        self.horizon = 1100

        self.state_dim = 3
        self.input_dim = 3

        self.lqr = InfiniteHorizonDiscreteLQR(A=self.A,
                                              B=self.B,
                                              Q=self.Q,
                                              R=self.R)

        A1 = self.lqr.A
        B1 = self.lqr.B
        F1 = self.lqr.F

        A2 = 2*self.A
        B2 = 2*self.B
        self.lqr.compute_dare_solution(A=A2, B=B2)

        self.assertTrue((np.abs(self.lqr.A - A1) < 1e-7).all())
        self.assertTrue((np.abs(self.lqr.B - B1) < 1e-7).all())
        self.assertTrue((np.abs(self.lqr.F - F1) > 1e-2).any())
        print('F1: \n', F1)
        print('New F: \n', self.lqr.F)

    def test_optimal_AB(self):
        """
        See if using the true A, B values actually leads to improvement in performance index.
        """
        self.A = np.array([[1.01, 0.01, 0],
                           [0.01, 1.01, 0.01],
                           [0, 0.01, 1.01]])
        self.B = np.identity(3)
        self.Q = 1e-3 * np.identity(3)
        self.R = np.identity(3)
        self.horizon = 1000  # 1_000_000

        self.state_dim = 3
        self.input_dim = 3

        self.lqr = InfiniteHorizonDiscreteLQR(A=self.A,
                                              B=self.B,
                                              Q=self.Q,
                                              R=self.R)

        J_opt, _ = self.lqr.simulate_trajectory(self.horizon)

        A2 = 1.5 * self.A
        B2 = 1.5 * self.B
        self.lqr.compute_dare_solution(A=A2, B=B2)
        J2, _ = self.lqr.simulate_trajectory(self.horizon)

        A3 = 0.5 * self.A
        B3 = 0.5 * self.B
        self.lqr.compute_dare_solution(A=A3, B=B3)
        J3, _ = self.lqr.simulate_trajectory(self.horizon)

        print('J_opt: ', J_opt)
        print('J2: ', J2)
        print('J3: ', J3)


class TestFiniteHorizonLQR(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_compute_correct_trajectory(self):
        """
        Ensure that the correct K_mats and k_vecs are produced, and that
        correct x1, u1, x2 are produced. Furthermore, ensure that u2 is NOT produced.
        """
        self.state_dim = 2
        self.input_dim = 1
        self.T = 2

        self.A = np.array([[2, 1],
                           [1, 1]])
        self.B = np.array([[1],
                           [1]])
        self.F = np.hstack((self.A, self.B))
        self.f = np.ones((2, 1))

        self.Q = np.identity(2)
        self.R = np.identity(1)
        self.C = np.vstack(
            (
                np.hstack((self.Q, np.zeros((2, 1)))),
                np.hstack((np.zeros((1, 2)), self.R))
            )
        )
        self.c = np.ones((3, 1))

        self.F_mats = np.repeat(self.F[:, :, np.newaxis], self.T, axis=2)
        self.f_vecs = np.squeeze(np.repeat(self.f[:, :, np.newaxis], self.T, axis=2))

        self.C_mats = np.repeat(self.C[:, :, np.newaxis], self.T, axis=2)
        self.c_vecs = np.squeeze(np.repeat(self.c[:, :, np.newaxis], self.T, axis=2))

        lqr = FiniteHorizonDiscreteLQR(C_mats=self.C_mats,
                                       c_vecs=self.c_vecs,
                                       F_mats=self.F_mats,
                                       f_vecs=self.f_vecs,
                                       T=self.T)
        K_mats, k_vecs = lqr.backward()

        # Computing the true K_mats and k_vecs values
        true_K_mat = np.array([[-1, -2 / 3]])
        true_k_vec = np.array([-5 / 3])

        self.assertTrue(K_mats.shape[2] == 1)
        self.assertTrue(k_vecs.shape[1] == 1)
        self.assertTrue((np.abs(K_mats[:, :, 0] - true_K_mat) < 1e-5).all())
        self.assertTrue((np.abs(k_vecs[:, 0] - true_k_vec) < 1e-5).all())

        # Testing sequence of states and control input
        x_init = np.array([1, 1])
        x_seq, u_seq = lqr.forward(x_init, K_mats, k_vecs)

        true_u1 = np.array([-10/3])
        true_x2 = np.array([2/3, -1/3])

        self.assertTrue(len(x_seq) == 2)
        self.assertTrue(len(u_seq) == 1)
        self.assertTrue((np.abs(x_seq[0] - x_init) < 1e-7).all())
        self.assertTrue((np.abs(x_seq[1] - true_x2) < 1e-7).all())
        self.assertTrue((np.abs(u_seq[0] - true_u1) < 1e-7).all())

    def test_finite_horizon_lqr(self):
        """
        Ensure that an FiniteHorizonDiscreteLQR object can be defined and run.
        """
        self.state_dim = 2
        self.input_dim = 1
        self.T = 10

        self.A = np.array([[-0.9, -0.3], [0.7, 0.1]])
        self.B = np.array([[1], [1]])

        self.F = np.hstack((self.A, self.B))
        self.f = np.array([[1], [1]])

        self.C = 10*np.identity(self.state_dim + self.input_dim)
        self.c = np.zeros((self.state_dim + self.input_dim, 1))

        self.F_mats = np.repeat(self.F[:, :, np.newaxis], self.T, axis=2)
        self.f_vecs = np.squeeze(np.repeat(self.f[:, :, np.newaxis], self.T, axis=2))

        self.C_mats = np.repeat(self.C[:, :, np.newaxis], self.T, axis=2)
        self.c_vecs = np.squeeze(np.repeat(self.c[:, :, np.newaxis], self.T, axis=2))

        self.x_init = np.array([1, 1])

        lqr = FiniteHorizonDiscreteLQR(C_mats=self.C_mats,
                                       c_vecs=self.c_vecs,
                                       F_mats=self.F_mats,
                                       f_vecs=self.f_vecs,
                                       T=self.T)
        K_mats, k_vecs = lqr.backward()
        x_seq, u_seq = lqr.forward(self.x_init, K_mats, k_vecs)
        print(x_seq)
