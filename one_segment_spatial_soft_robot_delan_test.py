import jax.numpy as jnp
import jax
from functools import partial
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pickle
import haiku as hk
import DeLaN_model_v4 as delan
import matlab
import matlab.engine

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

eng = matlab.engine.start_matlab()
### matlab_code:
eng.addpath(eng.genpath('/home/jing/Desktop/soft_robotics/matlab_code/one_segment/utils/'), nargout=0)
eng.addpath(eng.genpath('/home/jing/Desktop/soft_robotics/matlab_code/one_segment/funs/'), nargout=0)

mass_mat_fn = eng.B
# inv_mass_mat_fn = partial(eng.B_inv_fun, s)
coriolis_mat_fn = eng.C
gravity_fn = eng.G
input_mat_fn = eng.A
input_mat_inv_fn = eng.A_inv
dampping = np.array([[0.1, 0, 0], [0., 0.1, 0], [0., 0., 0.1]])
stiffness = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


## adjust the data type between matlab and numpy
def from_ml_to_np(ml_data):
    return np.array(ml_data._data).reshape(ml_data.size[::-1]).T

def from_np_to_ml(np_data):
    return matlab.double(np_data.tolist())


def equation_of_motion(state, u, t):
    q, dq = np.split(state, 2)
    q = q.reshape(q.shape[0], 1)
    dq = dq.reshape(dq.shape[0], 1)
    u = u.reshape(u.shape[0], 1)

    q = from_np_to_ml(q)
    dq = from_np_to_ml(dq)

    coriolis_mat = coriolis_mat_fn(q, dq)
    coriolis_mat = from_ml_to_np(coriolis_mat)
    grav = gravity_fn(q)
    grav = from_ml_to_np(grav)
    input_mat = input_mat_fn(q)
    input_mat = from_ml_to_np(input_mat)
    inv_mass_mat = eng.inv(mass_mat_fn(q))
    inv_mass_mat = from_ml_to_np(inv_mass_mat)
    # print(input_mat@input_inv_mat)
    ddq = inv_mass_mat @ (input_mat @ u - (coriolis_mat + dampping) @ dq - stiffness @ q - grav)
    return np.stack([dq[0], dq[1], dq[2], ddq[0], ddq[1], ddq[2]]).reshape(-1)


def rk4_step(f, x, y, t, h):
    # one step of runge-kutta integration
    k1 = h * f(x, y, t)
    # print(k1.shape)
    # print(x.shape)
    # print(x + k1/2)
    k2 = h * f(x + k1 / 2, y, t + h / 2)
    k3 = h * f(x + k2 / 2, y, t + h / 2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


total_time = 5
time_step = 0.01
# time_step = 0.01
t = np.arange(0, total_time, time_step)
n = t.shape[0]

# ###1.
init_state0 = np.array([1.6, -0.8, 0.07, 0.1, 0.01, 0.1])

u0 = np.ones(shape=(n, 3)) * 1.17 + np.random.random(size=(n, 3)) * 0.2
u0[100:120] += 0.9
u0[150:160] += 1.2
u0[180:, 0] += 0.5
u0[300:420] += 0.32
u0[450:470] += 0.3

###2.
# init_state0 = np.array([1.1, 0.6, 0.045, -0.1, 0.1, 0.1])
# u0 = - np.ones(shape=(n, 3)) * 0.7 + np.random.random(size=(n, 3)) * 0.8
# u0[200:230] -= 0.15
# u0[600:] += 0.65


###3.
# init_state0 = np.array([-1.5, -1.11, 0.055, -0.0, 0.0, 0.11])
# u0 = np.random.normal(size=(n, 3))*0.3 + np.ones(shape=(n, 3)) * 1.25
# u0[200:230] += 0.05

###4.
init_state0 = np.array([0.5, 0.5, 3.0, 0.01, 0.01, 0.1])
u0 = np.ones(shape=(n, 3)) * 1.17 + np.random.random(size=(n, 3)) * 0.4

state_transformation_fn = partial(rk4_step, equation_of_motion, t=0.0, h=time_step)



def data_generation(init_state, u):
    states = []
    targets = []
    state = init_state
    for i in range(n):
        states.append(state)
        state = state_transformation_fn(state, u[i])
        targets.append(state)
    states_real = np.array(states)
    return states_real

states_real = data_generation(init_state0, u0)

with open(f"./models/one_segment_spatial_soft_robot_delan.jax", 'rb') as f:
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
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

#dissipative_matrix(qd, n_dof, shape, activation)
dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(5,) * 3,
    activation=activations[hyper['activation2']]
))

#input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))



lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply

forward_model = jax.jit(delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat,
                                    input_mat=input_mat, n_dof=hyper['n_dof']))

state = init_state0
states =[]
for i in range(n):
    states.append(state)
    # print(u[i].shape)
    state = partial(rk4_step, forward_model, t=0.0, h=time_step)(state, u0[i])


states = np.array(states)


plt.subplot(2, 1, 1)
plt.grid()
plt.plot(t, states_real[:, 0], 'g', alpha=0.4, linewidth=3, label='$\Delta_x$')
plt.plot(t, states[:, 0], 'g', ls='--', linewidth=2, label='$\hat{\Delta_x}$')
plt.plot(t, states_real[:, 1], 'r', alpha=0.4,  linewidth=3, label='$\Delta_y$')
plt.plot(t, states[:, 1], 'r',  ls='--', linewidth=2, label='$\hat{\Delta_y}$')
plt.plot(t, states_real[:, 2], 'c', alpha=0.4,  linewidth=3, label='$\Delta_L$')
plt.plot(t, states[:, 2], 'c',  ls='--', linewidth=2, label='$\hat{\Delta_L}$')
plt.legend(fontsize=12, loc='upper right')
plt.ylabel("length [m]")
plt.xlabel('time [s]')
plt.yticks(size=12)
plt.xticks(size=12)
plt.subplot(2, 1, 2)
plt.grid()
plt.plot(t, states_real[:, 0] - states[:, 0], 'g', alpha=1, linewidth=2, label='$\Delta_x$')
plt.plot(t, states_real[:, 1] - states[:, 1], 'r', alpha=1,  linewidth=2, label='$\Delta y$')
plt.plot(t, states_real[:, 2] - states[:, 2], 'c', alpha=1,  linewidth=2, label='$\Delta_L$')
plt.legend(fontsize=12, loc='upper right')
plt.ylabel("length error [m]")
plt.xlabel('time [s]')
plt.yticks(size=12)
plt.xticks(size=12)

