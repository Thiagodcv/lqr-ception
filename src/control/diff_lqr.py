import numpy as np
from control.lqr import FiniteHorizonDiscreteLQR


class DifferentiableLQR:
    """
    An implementation of Differentiable LQR as outlined in Module 1 of
    "Differentiable MPC for End-to-end Planning and Control" (Amos et al., 2018).
    """

    def __init__(self, C_mats, c_vecs, F_mats, f_vecs, T, alpha):
        """
        The state space is n-dimensional and the control space is m-dimensional.

        :param C_mats: (n+m, n+m, T) array. PSD quadratic cost terms for all time steps t = 1,...,T.
        :param c_vecs: (n+m, T) array. Vector cost terms for all time steps t = 1,...,T.
        :param F_mats: (n, n+m, T) array. System dynamics matrix for all time steps t = 1,...,T.
        :param f_vecs: (n, T) array. System dynamics vector for all time steps t = 1,...,T.
        :param T: The horizon length. The number of nominal time steps to optimize for in the future.
        :param alpha: Step-size for gradient updates.
        """
        self.C_mats = C_mats
        self.c_vecs = c_vecs
        self.F_mats = F_mats
        self.f_vecs = f_vecs
        self.T = T
        self.alpha = alpha

        self.state_dim = self.F_mats.shape[0]
        self.input_dim = self.F_mats.shape[1] - self.state_dim

    def backward(self, tau_seq, lambda_seq, loss_grad):
        """
        The backward pass of the differentiable LQR algorithm.

        :param tau_seq: List containing optimal trajectory.
        :param lambda_seq: List of optimal lagrangian multipliers.
        :param loss_grad: Gradient of loss w.r.t optimal trajectory.
        :return: Derivatives of the loss function with respect to model parameters.
        """
        assert loss_grad.shape == self.c_vecs.shape
        lqr = FiniteHorizonDiscreteLQR(C_mats=self.C_mats,
                                       c_vecs=loss_grad,
                                       F_mats=self.F_mats,
                                       f_vecs=np.zeros(self.f_vecs.shape),
                                       T=self.T)
        K_mats, k_vecs = lqr.backward()
        d_x_seq, d_u_seq = lqr.forward(np.zeros(self.state_dim), K_mats, k_vecs)

        # Combine x_seq and u_seq to get optimal sequence d_seq
        d_tau_seq = []
        for t in range(self.T):
            d_tau_seq.append(np.hstack((d_x_seq[t], d_u_seq[t])))

        d_lambda_seq = self.compute_optimal_multipliers(d_tau_seq)

        grad = self.compute_gradients(tau_seq=tau_seq,
                                      d_tau_seq=d_tau_seq,
                                      lambda_seq=lambda_seq,
                                      d_lambda_seq=d_lambda_seq)
        return grad

    def forward(self, x_init):
        """
        The forward pass of the differentiable LQR algorithm.

        :param x_init: (n,) array. The initial state.
        :return: List of optimal trajectory, and list of optimal lagrangian multipliers.
        """
        lqr = FiniteHorizonDiscreteLQR(C_mats=self.C_mats,
                                       c_vecs=self.c_vecs,
                                       F_mats=self.F_mats,
                                       f_vecs=self.f_vecs,
                                       T=self.T)
        K_mats, k_vecs = lqr.backward()
        x_seq, u_seq = lqr.forward(x_init, K_mats, k_vecs)

        # Combine optimal state and input vectors into one optimal trajectory vector ''tau''
        tau_seq = []
        for t in range(self.T):
            tau_seq.append(np.hstack((x_seq[t], u_seq[t])))

        lambda_seq = self.compute_optimal_multipliers(tau_seq)
        return tau_seq, lambda_seq

    def compute_optimal_multipliers(self, tau_seq):
        """
        Computes the optimal lagrangian multipliers for equation (4) using equation (7).

        :param tau_seq: List of optimal trajectory vectors.
        :return: List of optimal lagrangian multipliers.
        """
        C_T_x = self.C_mats[:self.state_dim, :, -1]
        c_T_x = self.c_vecs[:self.state_dim, -1]
        lambda_seq = [C_T_x @ tau_seq[-1] + c_T_x]

        for t in reversed(range(self.T-1)):
            lambda_prev = lambda_seq[0]  # lambda from time t+1
            F_t_x = self.F_mats[:, :self.state_dim, t]
            C_t_x = self.C_mats[:self.state_dim, :, t]
            c_t_x = self.c_vecs[:self.state_dim, t]
            lambda_t = F_t_x.T @ lambda_prev + C_t_x @ tau_seq[t] + c_t_x
            lambda_seq = [lambda_t] + lambda_seq  # add lambda_t to beginning of list

        return lambda_seq

    def compute_gradients(self, tau_seq, d_tau_seq, lambda_seq, d_lambda_seq):
        """
        Computes gradients of loss function w.r.t LQR parameters.

        :param tau_seq: ?
        :param d_tau_seq: ?
        :param lambda_seq: ?
        :param d_lambda_seq: ?
        :return: dict containing gradients.
        """

        grad_C_mats = np.zeros(self.C_mats.shape)
        grad_c_vecs = np.zeros(self.c_vecs.shape)
        grad_F_mats = np.zeros(self.F_mats.shape)
        grad_f_vecs = np.zeros(self.f_vecs.shape)

        for t in range(self.T):
            # Gradient of loss w.r.t. C_t
            grad_C_mats[:, :, t] = 0.5*(np.outer(d_tau_seq[t], tau_seq[t]) + np.outer(tau_seq[t], d_tau_seq[t]))

            # Gradient of loss w.r.t. c_t
            grad_c_vecs[:, t] = d_tau_seq[t]

            # Gradient of loss w.r.t. F_t
            if t == self.T - 1:
                pass
            else:
                grad_F_mats[:, :, t] = np.outer(d_lambda_seq[t+1], tau_seq[t]) + np.outer(lambda_seq[t+1], d_tau_seq[t])

            # Gradient of loss w.r.t. f_t
            grad_f_vecs[:, t] = d_lambda_seq[t]

        # Gradient of loss w.r.t. x_init
        grad_x_init = d_lambda_seq[0]

        grad = {'C_mats': grad_C_mats,
                'c_vecs': grad_c_vecs,
                'F_mats': grad_F_mats,
                'f_vecs': grad_f_vecs,
                'x_init': grad_x_init}

        return grad

    def update_params(self, grad):
        """
        Use gradients to update LQR parameters. NOTE: Gradient update for x_init currently not implemented.

        :param grad: A dictionary containing the gradients of the LQR model.
        """
        self.C_mats -= self.alpha * grad['C_mats']
        self.c_vecs -= self.alpha * grad['c_vecs']
        self.F_mats -= self.alpha * grad['F_mats']
        self.f_vecs -= self.alpha * grad['f_vecs']
