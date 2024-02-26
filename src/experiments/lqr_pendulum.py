import numpy as np
import gymnasium as gym
from control_utils import compute_discrete_system
from control.lqr import FiniteHorizonDiscreteLQR
from environments.adjustable_pendulum import AdjustablePendulumEnv


def lqr_pendulum():
    """
    Because the dynamics at each timestep is the same linear system
    (i.e. the original nonlinear system linearized at the critical point), this algorithm runs
    into the same issue which MPC ran into, which is that it does not try to build up
    momentum in order to get around the upperbound on the control signal (the torque).
    :return:
    """
    g = 10
    max_speed = 8
    max_torque = 2
    init_state = {'th_init': 3/2 * np.pi, 'thdot_init': 2}
    seed = 10  # Note that if init_state is given, seed is overriden.

    env = AdjustablePendulumEnv(render_mode='human',
                                g=g,
                                max_speed=max_speed,
                                max_torque=max_torque,
                                init_state=init_state)

    # Dimensions of states and inputs
    state_dim = 2
    input_dim = 1

    # Horizon parameters
    T = 200
    dt = 0.005

    # Dynamics & Cost
    x_init = np.array([init_state['th_init'], init_state['thdot_init']])

    A = np.array([[0, 1],
                  [3 / 2 * g, 0]])
    B = np.array([[0], [3]])

    A_tilde, B_tilde = compute_discrete_system(A, B, dt)
    F = np.hstack((A_tilde, B_tilde))
    f = np.zeros((2, 1))

    # Cost function matrices
    Q = np.array([[500, 0],
                  [0, 1]])  # np.identity(state_dim)
    R = np.identity(input_dim)

    C = np.vstack(
        (
            np.hstack((Q, np.zeros((2, 1)))),
            np.hstack((np.zeros((1, 2)), R))
        )
    )
    c = np.zeros((3, 1))

    F_mats = np.repeat(F[:, :, np.newaxis], T, axis=2)
    f_vecs = np.squeeze(np.repeat(f[:, :, np.newaxis], T, axis=2))

    C_mats = np.repeat(C[:, :, np.newaxis], T, axis=2)
    c_vecs = np.squeeze(np.repeat(c[:, :, np.newaxis], T, axis=2))

    lqr = FiniteHorizonDiscreteLQR(C_mats=C_mats,
                                   c_vecs=c_vecs,
                                   F_mats=F_mats,
                                   f_vecs=f_vecs,
                                   T=T)
    K_mats, k_vecs = lqr.backward()
    x_seq, u_seq = lqr.forward(x_init, K_mats, k_vecs)
    #  print(x_seq)

    # Run environment
    env.reset()

    # Compute new trajectory at beginning of each time step (MPC-style LQR)
    # (This is helpful because x_k does not behave as LQR expects it to under
    # the linearized system).
    for k in range(T):
        action = u_seq[0].flatten()

        x_k, reward, terminated, truncated, info = env.step(action)

        x_seq, u_seq = lqr.forward(x_k, K_mats, k_vecs)

        if terminated or truncated:
            break

    # Compute trajectory at beginning only
    # for k in range(T):
    #     action = u_seq[k].flatten()
    #
    #     x_k, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated or truncated:
    #         break


if __name__ == '__main__':
    lqr_pendulum()
