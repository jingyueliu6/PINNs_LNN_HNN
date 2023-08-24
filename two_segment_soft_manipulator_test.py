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
#
def load_data():
    annots = loadmat(
        f"../data/test_data_two_segment/resample_data_1000Hz/simulation_test2.mat")
    q = annots["q"][:5000, :]
    dq = annots["dq"][:5000, :]
    tau = annots["tau"][:5000, :]
    # [:10001, :]
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
# with open(f"./models/two_segment_spatial_soft_robot_resample_data_50Hz.jax", "rb") as f:
# with open(f"./models/two_segment_spatial_soft_robot_resample_data_100Hz.jax", "rb") as f:
with open(f"./models/two_segment_spatial_soft_robot_resample_data_1000Hz.jax", "rb") as f:
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

# dissipative_matrix(qd, n_dof, shape, activation)
dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width_d'],) * hyper['n_depth_d'],
    activation=activations[hyper['activation2']]
))

# input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
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

forward_model = jax.jit(
    delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat,
                        input_mat=input_mat, n_dof=hyper['n_dof']))
# time_step = 0.0002 # for fixed time-step
# time_step = 0.01  # for variable time-step and resample frequency is 100
time_step = 0.001 # 1000Hz
# time_step = 0.02
### plan 3: every 30 state update the predicted state
state = states_real[0]
states = []
for i in range(n):
    states.append(state)
    state = partial(rk4_step, forward_model, t=0.0, h=time_step)(state, u[i])

states = np.array(states)
t = np.arange(0, time_step * n, time_step)

font1 = {'weight': 'normal',
         'size': 15,
         }
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

t = t[:n]


plt.subplot(2, 1, 1)
plt.plot(t, states_real[:n, 0], 'g', alpha=0.5, linewidth=2, label='$\Delta  x^{ref}$')
plt.plot(t, states[:, 0], 'g', ls='--', linewidth=2, label='$\Delta  x^{pred}$')
plt.plot(t, states_real[:n, 1], 'r', alpha=0.5,  linewidth=2, label='$\Delta  y^{ref}$')
plt.plot(t, states[:, 1], 'r',  ls='--', linewidth=2, label='$\Delta  y^{pred}$')
plt.plot(t, states_real[:n, 2], 'c', alpha=0.5,  linewidth=2, label='$\Delta  L^{ref}$')
plt.plot(t, states[:, 2], 'c',  ls='--', linewidth=2, label='$\Delta  L^{pred}$')
plt.legend(fontsize=15, loc='upper right')
# plt.ylabel("First segment",font1)
plt.xlabel('time[s]')
plt.xlabel('time[s]', font1)
plt.yticks(size=14)
plt.xticks(size=14)
#plt.title(f'Initial_State: delta_x({states_real[0]:.1e}), delta_y({states_real[1]:.1e}), dL({states_real[2]:.1e})', fontdict=font1)
plt.subplot(2, 1, 2)
plt.plot(t, states_real[:n, 3], 'm', alpha=0.5, linewidth=2, label='$\Delta  x^{ref}$')
plt.plot(t, states[:, 3], 'm', ls='--', linewidth=2, label='$\Delta  x^{pred}$')
plt.plot(t, states_real[:n, 4], 'b', alpha=0.5,  linewidth=2, label='$\Delta  y^{ref}$')
plt.plot(t, states[:, 4], 'b',  ls='--', linewidth=2, label='$\Delta  y^{pred}$')
plt.plot(t, states_real[:n, 5], 'purple', alpha=0.5,  linewidth=2, label='$\Delta  L^{ref}$')
plt.plot(t, states[:, 5], 'purple',  ls='--', linewidth=2, label='$\Delta  L^{pred}$')
plt.legend(fontsize=15, loc='upper right')
# plt.ylabel("Second segement",font1)
plt.xlabel('time[s]', font1)
plt.yticks(size=14)
plt.xticks(size=14)
# plt.title(f'Initial_Velocity: delta_x({states_real[3]:.1e}), delta_y({states_real[4]:.1e}) , dL({states_real[5]:.1e})', fontdict=font1)
plt.show()
accuracy_delan = np.sum((states - states_real)**2)/len(t)
print("accuracy_delan: ", accuracy_delan)

plt.subplot(1, 2, 1)
plt.plot(t, states_real[:n, 6], 'g', alpha=0.5, linewidth=2, label='segment1_real_delta_x_velocity')
plt.plot(t, states[:, 6], 'g', ls='--', linewidth=1, label='segment1_pred_delta_x_velocity')
plt.plot(t, states_real[:n, 7], 'r', alpha=0.5,  linewidth=2, label='segment1_real_delta_y_velocity')
plt.plot(t, states[:, 7], 'r',  ls='--', linewidth=1, label='segment1_pred_delta_y_velocity')
plt.plot(t, states_real[:n, 8], 'c', alpha=0.5,  linewidth=2, label='segment1_real_dL_velocity')
plt.plot(t, states[:, 8], 'c',  ls='--', linewidth=1, label='segment1_pred_dL_velocity')
plt.legend(fontsize=7, loc='upper right')
plt.ylabel("First segment")
plt.xlabel('time[s]')
#plt.title(f'Initial_State: delta_x({states_real[0]:.1e}), delta_y({states_real[1]:.1e}), dL({states_real[2]:.1e})', fontdict=font1)
plt.subplot(1, 2, 2)
plt.plot(t, states_real[:n, 9], 'm', alpha=0.5, linewidth=2, label='segment2_real_delta_x_velocity')
plt.plot(t, states[:, 9], 'm', ls='--', linewidth=1, label='segment2_pred_delta_x_velocity')
plt.plot(t, states_real[:n, 10], 'b', alpha=0.5,  linewidth=2, label='segment2_real_delta_y_velocity')
plt.plot(t, states[:, 10], 'b',  ls='--', linewidth=1, label='segment2_pred_delta_y_velocity')
plt.plot(t, states_real[:n, 11], 'purple', alpha=0.5,  linewidth=2, label='segment2_real_dL_velocity')
plt.plot(t, states[:, 11], 'purple',  ls='--', linewidth=1, label='segment2_pred_dL_velocity')
plt.legend(fontsize=7, loc='upper right')
plt.ylabel("Second segement")
plt.xlabel('time[s]')
#plt.title(f'Initial_Velocity: delta_x({states_real[3]:.1e}), delta_y({states_real[4]:.1e}) , dL({states_real[5]:.1e})', fontdict=font1)
plt.show()
