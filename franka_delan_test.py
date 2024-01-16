import jax.numpy as jnp
import jax
from functools import partial
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pickle
import haiku as hk
import DeLaN_model_v3 as delan
from scipy.io import loadmat
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_data():
    annots = loadmat(f"./data/franka_pybullet_test/dataset_test1.mat")
    q = annots["q"]
    dq = annots["dq"]
    tau = annots["input"]
    all_state = np.concatenate([q, dq], axis=1)
    states = all_state[:-1]
    targets = all_state[1:]
    u = tau[:-1]
    return states, targets, u


def rk4_step(f, x, y, t, h):
    # one step of runge-kutta integration
    k1 = h * f(x, y, t)
    k2 = h * f(x + k1 / 2, y, t + h / 2)
    k3 = h * f(x + k2 / 2, y, t + h / 2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


states_real, targets_real, u = load_data()
n = states_real.shape[0]
print(n)

with open(f"./models/franka_pybullet_1000Hz", "rb") as f:
    data = pickle.load(f)

hyper = data["hyper"]
params = data["params"]

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

#def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift)
lagrangian_fn = hk.transform(partial(
    delan.structured_lagrangian_fn,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width_l'],) * hyper['n_depth_l'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

#dissipative_matrix(qd, n_dof, shape, activation)
dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width_d'],) * hyper['n_depth_d'],
    activation=activations[hyper['activation2']]
))

lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply

forward_model = jax.jit(delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, n_dof=7))
time_step = 2.0e-3

state = states_real[0]
states =[]
steps = int(n)

print(steps)
for i in range(steps):
    states.append(state)
    # state = partial(rk4_step, forward_model, t=0.0, h=time_step)(states_real[i], u[i])
    state = partial(rk4_step, forward_model, t=0.0, h=time_step)(states[i], u[i])

states = np.array(states)
t = np.arange(0, 2.0e-3*n, 2.0e-3)
t_p = np.arange(0, time_step*steps, time_step)
font1 = {'weight': 'normal',
         'size': 11,
         }
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

error = jnp.sqrt(jnp.sum((states_real - states)**2))/n
print(error)
t = t[:n]
print(n)
print(t.shape)
print(states.shape)


plt.plot(t, states_real[:n, 0], 'g', alpha=0.4, linewidth=1, label=r'$j_1^{ref}$')
plt.plot(t, states[:, 0], 'g', ls='--', linewidth=1, label=r'$j_1^{grey}$')

plt.plot(t, states_real[:n, 1], 'r', alpha=0.4,  linewidth=1, label=r'$j_2^{ref}$')
plt.plot(t, states[:, 1], 'r',  ls='--', linewidth=1, label=r'$j_2^{grey}$')

plt.plot(t, states_real[:n, 2], 'c', alpha=0.4,  linewidth=1, label='$j_3^{ref}$')
plt.plot(t, states[:, 2], 'c',  ls='--', linewidth=1, label='$j_3^{grey}$')

plt.plot(t, states_real[:n, 3], 'k', alpha=0.4, linewidth=1, label=r'$j_4^{ref}$')
plt.plot(t, states[:, 3], 'k', ls='--', linewidth=1, label=r'$j_4^{grey}$')

plt.plot(t, states_real[:n, 4], 'm', alpha=0.4,  linewidth=1, label=r'$j_5^{ref}$')
plt.plot(t, states[:, 4], 'm',  ls='--', linewidth=1, label=r'$j_5^{grey}$')

plt.plot(t, states_real[:n, 5], 'y', alpha=0.4,  linewidth=1, label='$j_6^{ref}$')
plt.plot(t, states[:, 5], 'y',  ls='--', linewidth=1, label='$j_6^{grey}$')

plt.plot(t, states_real[:n, 6], 'purple', alpha=0.4, linewidth=1, label='$j_7^{ref}$')
plt.plot(t, states[:, 6], 'purple', ls='--', linewidth=1, label='$j_7^{grey}$')
plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle", font1)
plt.xlabel('time[s]', font1)
plt.yticks(size=11)
plt.xticks(size=11)
plt.show()
