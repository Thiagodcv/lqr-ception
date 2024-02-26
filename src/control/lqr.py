import numpy as np
import numbers


class InfiniteHorizonDiscreteLQR:
    """
    A simple implementation of tools needed to derive the F in the optimal control sequence u_t = -F * x_t,
    where u_t maximizes the LQR performance index used in the infinite-horizon, discrete-time setting.
    To learn more about the tools implemented in this class:
    https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator.
    """

    def __init__(self, A, B, Q, R):
        """
        Defines a discrete-time linear system x_{t+1} = A*x_t + B*u_t and a performance index J(u) parameterized
        by Q, R, and N.

        :param A: (state_dim x state_dim) array. As seen in the linear system.
        :param B: (state_dim x input_dim) array. As seen in the linear system.
        :param Q: (state_dim x state_dim) array. Penalizes state in performance index.
        :param R: (input_dim x input_dim) array. Penalizes control sequence in performance index.
        """
        # Ensure Q is positive semi-definite and R is positive definite, and that both are symmetric
        self.ensure_spd(Q, semi=True)
        self.ensure_spd(R, semi=False)

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.state_dim = self.B.shape[0]  # dimension of state variable
        self.input_dim = self.B.shape[1]  # dimension of control input

        # Unique positive definite solution to discrete time algebraic Riccati equation (DARE).
        self.P = None
        # Matrix for finding optimal control law.
        self.F = None
        # Computes P and F
        self.compute_dare_solution()

    def compute_dare_solution(self, A=None, B=None):
        """
        Computes unique positive definite solution to discrete time algebraic Riccati equation (DARE).
        :param A: (state_dim x state_dim) array. Used for overriding the true parameter A.
        :param B: (state_dim x input_dim) array. Used for overriding the true parameter A.
        """
        if A is None:
            A = self.A
        if B is None:
            B = self.B

        # Compute P
        curr_P = np.zeros(self.Q.shape)
        next_P = self.Q
        while (next_P - curr_P > 1e-7).any():
            curr_P = next_P
            next_P = self.Q + A.T @ curr_P @ A - A.T @ curr_P @ B @ \
                     np.linalg.inv(B.T @ curr_P @ B + self.R) @ B.T @ curr_P @ A

        self.P = next_P
        self.F = np.linalg.inv(self.R + B.T @ self.P @ B) @ (B.T @ self.P @ A)

    @staticmethod
    def ensure_spd(mat, semi):
        """
        If semi == True, throws an error if a matrix is not symmetric positive semi-definite.
        If semi == False, throws an error if a matrix is not symmetric positive definite.
        :param mat: A square numpy array
        :param semi: A boolean
        """
        if isinstance(mat, np.ndarray):
            if mat.shape[0] != mat.shape[1]:
                raise np.linalg.LinAlgError("Matrix not square.")
            if semi:
                np.linalg.cholesky(mat + 1e-7 * np.identity(mat.shape[0]))
            else:
                np.linalg.cholesky(mat)

            if (mat - mat.T > 1e-7).any():
                raise np.linalg.LinAlgError("Matrix not symmetric.")

        elif isinstance(mat, numbers.Number):
            if semi and mat < 0:
                raise np.linalg.LinAlgError("Matrix not positive semi-definite.")
            if not semi and mat <= 0:
                raise np.linalg.LinAlgError("Matrix not positive definite.")

    def simulate_trajectory(self, horizon, x0=None):
        """
        Simulates the discrete-time linear system for ``horizon'' steps while using the
        optimal control sequence for LQR.

        :param horizon: An int representing the number of time steps to run the trajectory for.
        :param x0: (state_dim,) array representing initial state.
        :return: A float representing the value of the performance index under a finite horizon, and
                 a list containing the norm difference of subsequent states.
        """
        J = 0
        diffs = []
        if x0 is not None:
            x_t = x0
        else:
            x_t = np.random.standard_normal((self.state_dim,))

        for t in range(horizon):
            u_t = -self.F @ x_t
            J += (x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t)
            x_next = self.A @ x_t + self.B @ u_t
            diffs.append(np.linalg.norm(x_next - x_t))
            x_t = x_next

        return J, diffs


class FiniteHorizonDiscreteLQR:
    """
    An implementation of discrete-time finite-horizon LQR as outlined in appendix A of
    "Differentiable MPC for End-to-end Planning and Control" (Amos et al., 2018).
    """

    def __init__(self, C_mats, c_vecs, F_mats, f_vecs, T):
        """
        The state space is n-dimensional and the control space is m-dimensional.

        :param C_mats: (n+m, n+m, T) array. PSD quadratic cost terms for all time steps t = 1,...,T.
        :param c_vecs: (n+m, T) array. Vector cost terms for all time steps t = 1,...,T.
        :param F_mats: (n, n+m, T) array. System dynamics matrix for all time steps t = 1,...,T.
        :param f_vecs: (n, T) array. System dynamics vector for all time steps t = 1,...,T.
        :param T: The horizon length. The number of nominal time steps to optimize for in the future.
        """
        self.C_mats = C_mats
        self.c_vecs = c_vecs
        self.F_mats = F_mats
        self.f_vecs = f_vecs
        self.T = T

        self.state_dim = self.F_mats.shape[0]
        self.input_dim = self.F_mats.shape[1] - self.state_dim

    def backward(self):
        """
        Compute optimal LQR control feedback for each timestep.

        NOTE: Only the C_T_xx and c_t_x portions of C_T and c_T will be used, since this algorithm
        only computes optimal u_1,...,u_{T-1} and not u_T.

        :return: Two arrays used for computing the optimal LQR control feedback at each timestep.
        """
        V_mats = np.zeros((self.state_dim, self.state_dim, self.T))
        v_vecs = np.zeros((self.state_dim, self.T))

        K_mats = np.zeros((self.input_dim, self.state_dim, self.T-1))
        k_vecs = np.zeros((self.input_dim, self.T-1))

        V_mats[:, :, self.T-1] = self.C_mats[:self.state_dim, :self.state_dim, self.T-1]
        v_vecs[:, self.T-1] = self.c_vecs[:self.state_dim, self.T-1]

        for t in reversed(range(self.T-1)):
            Q_t = self.C_mats[:, :, t] + self.F_mats[:, :, t].T @ V_mats[:, :, t+1] @ self.F_mats[:, :, t]
            q_t = self.c_vecs[:, t] + \
                  self.F_mats[:, :, t].T @ V_mats[:, :, t+1] @ self.f_vecs[:, t] + \
                  self.F_mats[:, :, t].T @ v_vecs[:, t+1]

            Q_t_xx = Q_t[:self.state_dim, :self.state_dim]
            Q_t_ux = Q_t[self.state_dim:, :self.state_dim]
            Q_t_xu = Q_t[:self.state_dim, self.state_dim:]
            Q_t_uu = Q_t[self.state_dim:, self.state_dim:]
            Q_t_uu_inv = np.linalg.inv(Q_t_uu)

            q_t_x = q_t[:self.state_dim]
            q_t_u = q_t[self.state_dim:]

            K_mats[:, :, t] = -Q_t_uu_inv @ Q_t_ux
            k_vecs[:, t] = -Q_t_uu_inv @ q_t_u

            V_mats[:, :, t] = Q_t_xx - K_mats[:, :, t].T @ Q_t_uu @ K_mats[:, :, t]
            v_vecs[:, t] = q_t_x + Q_t_xu @ k_vecs[:, t]

        return K_mats, k_vecs

    def forward(self, x_init, K_mats, k_vecs):
        """
        Update state and control input trajectory based on the control feedback calculated during the backwards pass.

        :param x_init: (n,) array. The initial state.
        :param K_mats: (m, n, T) array. Control feedback matrices for time steps t=1,...,T.
        :param k_vecs: (m, T) array. Control feedback vectors for time steps t=1,...,T.
        :return: List containing state trajectory, list containing control input trajectory.
        """
        x_t = x_init

        x_seq = [x_t]
        u_seq = []

        for t in range(self.T-1):
            u_t = K_mats[:, :, t] @ x_t + k_vecs[:, t]
            u_seq.append(u_t)
            x_t = self.F_mats[:, :, t] @ np.hstack((x_t, u_t)) + self.f_vecs[:, t]
            x_seq.append(x_t)

        return x_seq, u_seq
