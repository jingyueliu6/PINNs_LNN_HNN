import jax.numpy as jnp
import jax
from functools import partial
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pickle
import haiku as hk
import DeLaN_model_v4 as delan
from scipy.io import loadmat
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_data():
    annots = loadmat(f".data/dataset_test3.mat")
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
with open(f"./models/real_one_cable_5Hz.jax", "rb") as f:
    data = pickle.load(f)

hyper = data["hyper"]
params = data["params"]

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

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

#input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width_i'],) * hyper['n_depth_i'],
    activation=activations[hyper['activation1']]
))

lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply

forward_model = jax.jit(delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat,
                                    input_mat=input_mat, n_dof=hyper['n_dof']))

time_step = 0.2


state = states_real[0]
states =[]
for i in range(n):
    states.append(state)
    state = partial(rk4_step, forward_model, t=0.0, h=time_step)(states_real[i], u[i])
    #state = partial(rk4_step, forward_model, t=0.0, h=time_step)(state, u[i])



states = np.array(states)
t = np.arange(0, time_step*n, time_step)
font1 = {'weight': 'normal',
         'size': 24,
         }
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
np.savetxt("./data_2_figure/one_segment_real_fitting.csv", states_real, delimiter=",")
np.savetxt("./data_2_figure/one_segment_pred_fitting.csv", states, delimiter=",")

t = t[:n]
print(n)
print(t.shape)
print(states.shape)
plt.subplot(1, 1, 1)
plt.plot(t, states_real[:n, 0], 'g', alpha=0.4, linewidth=3, label=r'$\alpha^{ref}$')
plt.plot(t, states[:, 0], 'g', ls='--', linewidth=3, label=r'$\alpha^{grey}$')
plt.plot(t, states_real[:n, 1], 'r', alpha=0.4,  linewidth=3, label=r'$\beta^{ref}$')
plt.plot(t, states[:, 1], 'r',  ls='--', linewidth=3, label=r'$\beta^{grey}$')
plt.plot(t, states_real[:n, 2], 'c', alpha=0.4,  linewidth=3, label='$\gamma^{ref}$')
plt.plot(t, states[:, 2], 'c',  ls='--', linewidth=3, label='$\gamma^{grey}$')
plt.legend(fontsize=24, loc='upper right')
plt.ylabel("Orientation", font1)
plt.xlabel('time[s]', font1)
plt.yticks(size=17)
plt.xticks(size=17)
plt.show()


accuracy_delan = np.sum((states - states_real)**2)/len(t)
print("accuracy_delan: ", accuracy_delan)


