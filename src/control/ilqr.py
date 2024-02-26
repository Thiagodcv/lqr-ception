"""
Credits for this code mainly goes to Shuyuan Wang: https://dais.chbe.ubc.ca/people/shuyuan-wang/.
I simply cleaned up the code (e.g. removed redundant operations and made it more readable),
and made it more consistent with OOP practices.
"""

from autograd import grad, jacobian
import autograd.numpy as np


class ILQR:
    """
    An implementation of the Iterative Linear Quadratic Regulator algorithm, based off of
    UC Berkeley CS 285: Lecture 10, Part 4:
    https://www.youtube.com/watch?v=PHC2dm4E_VQ&list=PL_iWQOsE6TfX7MaC6C3HcdOf1g337dlC9&index=46.
    """

    def __init__(self, dynamics_func, running_cost, terminal_cost, horizon, state_dim, input_dim):
        """
        :param dynamics_func: A lambda function representing the deterministic dynamics of a system.
        :param running_cost: A lambda function representing the running cost of a trajectory of a system.
        :param terminal_cost: A lambda function representing the cost of the terminal state.
        :param horizon: (int) Number of timesteps into the future to compute trajectory.
        :param state_dim: (int) Dimension of the state vector x.
        :param input_dim: (int) Dimension of the input vector u.
        """
        self.f = dynamics_func
        self.c = running_cost
        self.ct = terminal_cost
        self.horizon = horizon
        self.state_dim = state_dim
        self.input_dim = input_dim

        # Jacobians and Hessians of the dynamics used for linearization
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)

        self.f_xx = jacobian(self.f_x, 0)
        self.f_ux = jacobian(self.f_u, 0)
        self.f_uu = jacobian(self.f_u, 1)

        # Jacobians and Hessians of the running cost function used for linearization
        self.c_x = grad(self.c, 0)
        self.c_u = grad(self.c, 1)

        self.c_xx = jacobian(self.c_x, 0)
        self.c_ux = jacobian(self.c_u, 0)
        self.c_uu = jacobian(self.c_u, 1)

        # Jacobian and Hessian of the terminal cost
        self.ct_x = grad(self.ct, 0)
        self.ct_xx = jacobian(self.ct_x, 0)

    def backwards(self, x_seq, u_seq):
        """
        Compute optimal LQR control feedback for each timestep.
        :param x_seq: List of states.
        :param u_seq: List of control inputs.
        :return: A list containing the optimal LQR control feedback for each timestep.
        """
        k_vec_seq = []
        K_mat_seq = []

        # Why initialize these to be Jacobians of terminal cost?
        v_x_t = self.ct_x(x_seq[-1], u_seq[-1])
        v_xx_t = self.ct_xx(x_seq[-1], u_seq[-1])

        for t in reversed(range(self.horizon)):
            x = x_seq[t]
            u = u_seq[t]

            # Evaluate Jacobians
            f_x_t = self.f_x(x, u)
            f_u_t = self.f_u(x, u)

            f_xx_t = self.f_xx(x, u)
            f_uu_t = self.f_uu(x, u)
            f_ux_t = self.f_ux(x, u)

            c_x_t = self.c_x(x, u)
            c_u_t = self.c_u(x, u)

            c_xx_t = self.c_xx(x, u)
            c_uu_t = self.c_uu(x, u)
            c_ux_t = self.c_ux(x, u)

            # Compute matrices for state-action value functions
            q_x = c_x_t + f_x_t.T @ v_x_t
            q_u = c_u_t + f_u_t.T @ v_x_t

            q_xx = c_xx_t + f_x_t.T @ v_xx_t @ f_x_t + (v_x_t.reshape(-1, 1, 1) * f_xx_t).sum(axis=0)
            q_uu = c_uu_t + f_u_t.T @ v_xx_t @ f_u_t + (v_x_t.reshape(-1, 1, 1) * f_uu_t).sum(axis=0)
            q_ux = c_ux_t + f_u_t.T @ v_xx_t @ f_x_t + (v_x_t.reshape(-1, 1, 1) * f_ux_t).sum(axis=0)

            # Invert with some regularization
            q_uu_inv = np.linalg.inv(q_uu + 1e-9 * np.eye(q_uu.shape[0]))

            # Compute matrices for optimal control input
            k_vec = -q_uu_inv.dot(q_u)  # previously called k
            K_mat = -q_uu_inv.dot(q_ux)  # previously called kk

            # Compute matrices for value functions
            v_x_t = q_x + K_mat.T @ q_uu @ k_vec + K_mat.T @ q_u + q_ux.T @ k_vec
            v_xx_t = q_xx + K_mat.T @ q_uu @ K_mat + K_mat.T @ q_ux + q_ux.T @ K_mat

            k_vec_seq.append(k_vec)
            K_mat_seq.append(K_mat)

        k_vec_seq.reverse()
        K_mat_seq.reverse()

        return k_vec_seq, K_mat_seq

    def forward(self, x_seq, u_seq, k_vec_seq, K_mat_seq, alpha):
        """
        Update state and control input trajectory based on the control feedback calculated during the backwards pass.

        :param x_seq: List of states.
        :param u_seq: List of control inputs.
        :param k_vec_seq: List of arrays used for calculating optimal control input at each timestep (until horizon).
        :param K_mat_seq: List of arrays used for calculating optimal control input at each timestep (until horizon).
        :param alpha: Backtracking line-search hyperparameter. TODO: Do more research on alpha.
        :return: List containing state trajectory, list containing control input trajectory.

        TODO: Figure out how to handle constraints on x and u?
        """
        x_seq_new = np.array(x_seq)
        u_seq_new = np.array(u_seq)

        for t in range(len(u_seq)):
            du = K_mat_seq[t] @ (x_seq_new[t] - x_seq[t]) + alpha**t * k_vec_seq[t]
            u_seq_new[t] = u_seq[t] + du
            x_seq_new[t+1] = self.f(x_seq_new[t], u_seq_new[t])

        return x_seq_new, u_seq_new

    def simulate_env(self, env, env_horizon, back_forward_reps=3, alpha=0.99,
                     seed=None, mod_obs=None, mod_action=None, verbose=False):
        """
        Simulates ILQR control of a given gym environment.

        :param env: The (gym) environment to simulate.
        :param env_horizon: Number of time steps to simulate the environment for.
        :param back_forward_reps: (int) Number of times to recompute optimal trajectory per time step.
        :param alpha: (float) Number in between (0, 1). For the line-search in the forward pass.
        :param seed: The seed for the initial state.
        :param mod_obs: A function for modifying the state output from the environment so that
                        they match the dynamics specified in self.dynamics_func.
        :param mod_action: A function for modifying the actions output by ILQR right before
                           they are input into the environment env.
        :param verbose: If true, prints the state and control_input at each timestep.
        :return: A list of floats (the rewards produced at each timestep of the MDP).
        """
        rewards = []

        if seed is None:
            x_k, info = env.reset()
        else:
            x_k, info = env.reset(seed=seed)
        u_seq = [np.zeros(self.input_dim) for _ in range(self.horizon)]

        for k in range(env_horizon):
            if mod_obs is not None:
                x_k = mod_obs(x_k)

            # Rollout a trajectory of the nonlinear system under current control sequence
            x_seq = [x_k]
            for t in range(self.horizon):
                x_seq.append(self.f(x_seq[-1], u_seq[t]))

            # Compute new 'optimal' state and input trajectory
            for _ in range(back_forward_reps):
                k_vec_seq, K_mat_seq = self.backwards(x_seq, u_seq)
                x_seq, u_seq = self.forward(x_seq, u_seq, k_vec_seq, K_mat_seq, alpha=alpha)

            if verbose:
                print('--------------')
                print('k: ', k)
                print('x_k: \n', x_k)
                print('u_k: \n', u_seq[0])

            if mod_action is not None:
                action = mod_action(u_seq[0])
                x_k, reward, terminated, truncated, info = env.step(action)
            else:
                x_k, reward, terminated, truncated, info = env.step(u_seq[0])

            rewards.append(reward)

            if terminated or truncated:
                break

        env.close()
        return rewards
