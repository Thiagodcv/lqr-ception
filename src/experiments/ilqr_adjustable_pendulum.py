import autograd.numpy as np
from control.ilqr import ILQR
from environments.adjustable_pendulum import AdjustablePendulumEnv


def ilqr_adjustable_pendulum():
    g = 10
    max_speed = 8
    max_torque = 2
    init_state = {'th_init': 1.1*np.pi, 'thdot_init': 2}
    seed = None  # Note that if init_state is given, seed is overriden.

    env = AdjustablePendulumEnv(render_mode='human',
                                g=g,
                                max_speed=max_speed,
                                max_torque=max_torque,
                                init_state=init_state)

    ilqr = ILQR(dynamics_func=lambda x, u: pendulum_dynamics(x, u),
                running_cost=lambda x, u: cost(x, u),
                terminal_cost=lambda x, u: cost(x, u),
                horizon=20,
                state_dim=2,
                input_dim=1)

    ilqr.simulate_env(env=env,
                      env_horizon=200,
                      seed=seed,
                      back_forward_reps=1,
                      mod_obs=None,
                      mod_action=None,
                      verbose=False)


def cost(x, u):
    costs = angle_normalize(x[0]) ** 2 + .1 * x[1] ** 2 + .001 * (u ** 2)
    return costs


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


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


if __name__ == '__main__':
    ilqr_adjustable_pendulum()
