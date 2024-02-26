from unittest import TestCase
from control.diff_lqr import DifferentiableLQR
from control.lqr import FiniteHorizonDiscreteLQR
import numpy as np


class TestDifferentiableLQR(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_gradient(self):
        """
        Compare computed gradient with gradient obtained from finite-differencing.
        TODO: Fix this test.
        """
        self.state_dim = 2
        self.input_dim = 1
        self.T = 2
        self.alpha = 1e-5

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

        # Define LQRs
        C_mats_diff = np.random.standard_normal(self.C_mats.shape)
        c_vecs_diff = np.random.standard_normal(self.c_vecs.shape)
        F_mats_diff = np.random.standard_normal(self.F_mats.shape)
        f_vecs_diff = np.random.standard_normal(self.f_vecs.shape)

        # lqr_diff = DifferentiableLQR(C_mats=C_mats_diff,
        #                              c_vecs=c_vecs_diff,
        #                              F_mats=F_mats_diff,
        #                              f_vecs=f_vecs_diff,
        #                              T=self.T,
        #                              alpha=self.alpha)

        # perturb = np.zeros(self.C_mats.shape)
        # perturb[2, 2, 0] = -1e-5
        # C_mats_perturbed = self.C_mats + perturb

        lqr_diff = DifferentiableLQR(C_mats=self.C_mats,
                                     c_vecs=self.c_vecs,
                                     F_mats=self.F_mats,
                                     f_vecs=self.f_vecs,
                                     T=self.T,
                                     alpha=self.alpha)

        lqr_true = FiniteHorizonDiscreteLQR(C_mats=self.C_mats,
                                            c_vecs=self.c_vecs,
                                            F_mats=self.F_mats,
                                            f_vecs=self.f_vecs,
                                            T=self.T)

        x_init = np.array([1, 1])

        # Compute trajectory from lqr_true (needed for loss function)
        K_mats, k_vecs = lqr_true.backward()
        x_seq_true, u_seq_true = lqr_true.forward(x_init, K_mats, k_vecs)
        tau_true = np.hstack((x_seq_true[0], u_seq_true[0]))
        for i in range(1, self.T):
            new_rows = np.hstack((x_seq_true[i], u_seq_true[i]))
            tau_true = np.vstack((tau_true, new_rows))

        # Compute trajectory from lqr_diff
        tau_seq_diff, lambda_seq_diff = lqr_diff.forward(x_init)
        tau_diff = tau_seq_diff[0]
        for j in range(1, self.T):
            tau_diff = np.vstack((tau_diff, tau_seq_diff[j]))

        # Compute the gradient of the loss in the shape of c_vecs
        # loss_grad = tau_true - tau_diff
        loss_grad = tau_diff - tau_true

        # Compute gradients of loss w.r.t parameters of lqr_diff
        grad = lqr_diff.backward(tau_seq_diff, lambda_seq_diff, loss_grad.T)

        # Finite differencing step
        dt = 1e-7  # 0.33307 * 1e-15

        perturb = np.zeros(self.C_mats.shape)
        perturb[2, 2, 0] = dt
        # C_mats_diff_perturbed = C_mats_diff + perturb

        # lqr_diff_perturbed = DifferentiableLQR(C_mats=C_mats_diff_perturbed,
        #                                        c_vecs=c_vecs_diff,
        #                                        F_mats=F_mats_diff,
        #                                        f_vecs=f_vecs_diff,
        #                                        T=self.T,
        #                                        alpha=self.alpha)

        C_mats_perturbed = self.C_mats + perturb

        lqr_diff_perturbed = DifferentiableLQR(C_mats=C_mats_perturbed,
                                               c_vecs=self.c_vecs,
                                               F_mats=self.F_mats,
                                               f_vecs=self.f_vecs,
                                               T=self.T,
                                               alpha=self.alpha)

        # Compute optimal trajectory of perturbed lqr
        tau_seq_diff_perturbed, _ = lqr_diff_perturbed.forward(x_init)
        tau_diff_perturbed = tau_seq_diff_perturbed[0]
        for j in range(1, self.T):
            tau_diff_perturbed = np.vstack((tau_diff_perturbed, tau_seq_diff_perturbed[j]))

        loss = np.linalg.norm(tau_true - tau_diff)
        perturbed_loss = np.linalg.norm(tau_true - tau_diff_perturbed)
        finite_diff_grad = (perturbed_loss - loss) / dt

        print("Analytical gradient: ", grad['C_mats'][2, 2, 0])
        print("Finite difference gradient: ", finite_diff_grad)
        print("C_mats_perturbed - C_mats Norm: ", np.linalg.norm(C_mats_perturbed - self.C_mats))

    def test_compute_gradients(self):
        """
        Ensure the compute_gradients() method returns the correct output.
        """
        self.state_dim = 2
        self.input_dim = 1
        self.T = 2
        self.alpha = 1e-5

        self.C_mats = np.zeros((3, 3, 2))
        self.c_vecs = np.zeros((3, 2))

        self.F_mats = np.zeros((2, 3, 2))
        self.f_vecs = np.zeros((2, 2))

        lqr_diff = DifferentiableLQR(C_mats=self.C_mats,
                                     c_vecs=self.c_vecs,
                                     F_mats=self.F_mats,
                                     f_vecs=self.f_vecs,
                                     T=self.T,
                                     alpha=self.alpha)

        d_tau_1 = np.ones(3)
        d_lambda_1 = 2 * np.ones(2)
        tau_1 = 3 * np.ones(3)
        lambda_1 = 4 * np.ones(2)

        d_tau_2 = 4 * np.ones(3)
        d_lambda_2 = 3 * np.ones(2)
        tau_2 = 2 * np.ones(3)
        lambda_2 = np.ones(2)

        d_tau_seq = [d_tau_1, d_tau_2]
        d_lambda_seq = [d_lambda_1, d_lambda_2]
        tau_seq = [tau_1, tau_2]
        lambda_seq = [lambda_1, lambda_2]

        grad = lqr_diff.compute_gradients(tau_seq=tau_seq,
                                          d_tau_seq=d_tau_seq,
                                          lambda_seq=lambda_seq,
                                          d_lambda_seq=d_lambda_seq)

        grad_C_1 = 3 * np.ones(self.C_mats.shape)[:, :, 0]
        grad_F_1 = 10 * np.ones(self.F_mats.shape)[:, :, 0]
        grad_c_1 = 1 * np.ones(self.c_vecs.shape)[:, 0]
        grad_f_1 = 2 * np.ones(self.f_vecs.shape)[:, 0]

        grad_C_2 = 8 * np.ones(self.C_mats.shape)[:, :, 0]
        grad_F_2 = np.zeros(self.F_mats.shape)[:, :, 0]
        grad_c_2 = 4 * np.ones(self.c_vecs.shape)[:, 0]
        grad_f_2 = 3 * np.ones(self.f_vecs.shape)[:, 0]

        self.assertTrue((np.abs(grad['C_mats'][:, :, 0] - grad_C_1) < 1e-7).all())
        self.assertTrue((np.abs(grad['F_mats'][:, :, 0] - grad_F_1) < 1e-7).all())
        self.assertTrue((np.abs(grad['c_vecs'][:, 0] - grad_c_1) < 1e-7).all())
        self.assertTrue((np.abs(grad['f_vecs'][:, 0] - grad_f_1) < 1e-7).all())

        self.assertTrue((np.abs(grad['C_mats'][:, :, 1] - grad_C_2) < 1e-7).all())
        self.assertTrue((np.abs(grad['F_mats'][:, :, 1] - grad_F_2) < 1e-7).all())
        self.assertTrue((np.abs(grad['c_vecs'][:, 1] - grad_c_2) < 1e-7).all())
        self.assertTrue((np.abs(grad['f_vecs'][:, 1] - grad_f_2) < 1e-7).all())

    def test_compute_optimal_multipliers(self):
        self.state_dim = 2
        self.input_dim = 1
        self.T = 2
        self.alpha = 1e-5

        self.A = np.array([[2, 1],
                           [1, 1]])
        self.B = np.array([[1],
                           [1]])
        self.F = np.hstack((self.A, self.B))
        self.f = np.zeros((2, 1))

        self.C = np.identity(3)
        self.c = np.ones((3, 1))

        self.F_mats = np.repeat(self.F[:, :, np.newaxis], self.T, axis=2)
        self.f_vecs = np.squeeze(np.repeat(self.f[:, :, np.newaxis], self.T, axis=2))

        self.C_mats = np.repeat(self.C[:, :, np.newaxis], self.T, axis=2)
        self.c_vecs = np.squeeze(np.repeat(self.c[:, :, np.newaxis], self.T, axis=2))

        lqr_diff = DifferentiableLQR(C_mats=self.C_mats,
                                     c_vecs=self.c_vecs,
                                     F_mats=self.F_mats,
                                     f_vecs=self.f_vecs,
                                     T=self.T,
                                     alpha=self.alpha)

        tau_1 = np.ones(3)
        tau_2 = 2 * np.ones(3)
        tau_seq = [tau_1, tau_2]

        lambda_1_true = np.array([11, 8])
        lambda_2_true = np.array([3, 3])
        lambda_seq = lqr_diff.compute_optimal_multipliers(tau_seq)

        self.assertTrue((np.abs(lambda_seq[0] - lambda_1_true) < 1e-7).all())
        self.assertTrue((np.abs(lambda_seq[1] - lambda_2_true) < 1e-7).all())
