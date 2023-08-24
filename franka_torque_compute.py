import time
import math

from panda import Panda ## find a pybullet panda example, for the trained model, the panda do not have the gripper
import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pickle
import haiku as hk
import DeLaN_model_v3 as delan

Kp = [100.0, 100.0, 100.0, 150.0, 20.0, 12.0, 20.0]
Kd = [10.0, 10.0, 10.0, 3.0, 1.0, 2.5, 0.5]

duration = 20
stepsize = 1e-3

robot = Panda(stepsize)
robot.setControlMode("torque")
t = 0.
pos, vel = robot.getJointStates()
acc = [0 for x in pos]
pos_exp = pos
vel_exp = vel

with open(f"./franka_attempt_1000Hz.jax", "rb") as f:
    data = pickle.load(f)

hyper = data["hyper"]
params = data["params"]

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

#def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift)conda
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
    activation=activations[hyper['activation1']]
))

lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply

inverse_model = jax.jit(delan.inverse_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, n_dof=7))
state = []
ts = []
torques = []

for i in range(int(duration / stepsize)):
    if i % 1000 == 0:
        print("Simulation time: {:.3f}".format(robot.t))
        print("pos: ", pos)
    acc_exp = [0 for x in pos]
    pos_exp[0] = -0.3 * math.sin(2 * math.pi * 0.2 * t)
    pos_exp[1] = 0.1 * math.sin(2 * math.pi * 0.1 * t)
    pos_exp[2] = -0.3 * math.sin(2 * math.pi * 0.1 * t)
    pos_exp[3] = -0.5 * math.sin(2 * math.pi * 0.2 * t) - 1.0
    pos_exp[4] = -0.1 * math.sin(2 * math.pi * 0.3 * t)
    pos_exp[5] = 1.0* math.sin(2 * math.pi * 0.3 * t) + 1.0
    pos_exp[6] = -0.05 * math.sin(2 * math.pi * 0.3 * t)

    vel_exp[0] = -0.3 * 2 * math.pi * 0.2 *  math.cos(2 * math.pi * 0.2 * t)
    vel_exp[1] = 0.1 * 2 * math.pi * 0.1 *  math.cos(2 * math.pi * 0.1 * t)
    vel_exp[2] = -0.3 * 2 * math.pi * 0.1 *  math.cos(2 * math.pi * 0.1 * t)
    vel_exp[3] = -0.5 * 2 * math.pi * 0.2 *  math.cos(2 * math.pi * 0.2 * t) - 1.0
    vel_exp[4] = -0.1 * 2 * math.pi * 0.3 *  math.cos(2 * math.pi * 0.3 * t)
    vel_exp[5] = 1.0 * 2 * math.pi * 0.3 * math.cos(2 * math.pi * 0.3 * t)
    vel_exp[6] = -0.05 * 2 * math.pi * 0.3 * math.cos(2 * math.pi * 0.3 * t)

    torque = inverse_model(np.concatenate([pos, vel], axis=0), np.array(acc_exp))

    torques.append(torque)
    target_torque = Kp * (np.array(pos_exp) - np.array(pos)) + Kd * (np.array(vel_exp) - np.array(vel)) + torque

    pos, vel, = robot.getJointStates()
    robot.setTargetTorques(target_torque)
    tor = target_torque
    current_torque = tor

    sample = [t, tor[0], tor[1], tor[2], tor[3], tor[4], tor[5], tor[6],
              pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6],
              vel[0], vel[1], vel[2], vel[3], vel[4], vel[5], vel[6]]

    state.append([pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]])
    pos, vel, = robot.getJointStates()
    ts.append(t)
    t += stepsize

    robot.step()

    time.sleep(robot.stepsize)

state = np.array(state)
torques = np.array(torques)


plt.subplot(7, 1, 1)
plt.plot(ts, state[:, 0], 'g', alpha=0.4, linewidth=1, label=r'$j_1$')
plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle")
plt.xlabel('time [s]')
plt.subplot(7, 1, 2)
plt.plot(ts, state[:, 1], 'r', alpha=0.4,  linewidth=1, label=r'$j_2$')
plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle")
plt.xlabel('time [s]')
plt.subplot(7, 1, 3)
plt.plot(ts, state[:, 2], 'c', alpha=0.4,  linewidth=1, label='$j_3$')

plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle")
plt.xlabel('time [s]')
plt.subplot(7, 1, 4)
plt.plot(ts, state[:, 3], 'k', alpha=0.4, linewidth=1, label=r'$j_4$')
plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle")
plt.xlabel('time [s]')
plt.subplot(7, 1, 5)
plt.plot(ts, state[:, 4], 'm', alpha=0.4,  linewidth=1, label=r'$j_5$')
plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle")
plt.xlabel('time [s]')
plt.subplot(7, 1, 6)
plt.plot(ts, state[:, 5], 'y', alpha=0.4,  linewidth=1, label='$j_6$')
plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle")
plt.xlabel('time [s]')
plt.subplot(7, 1, 7)
plt.plot(ts, state[:, 6], 'purple', alpha=0.4, linewidth=1, label='$j_7}$')
plt.legend(fontsize=11, loc='upper right')
plt.ylabel("angle")
plt.xlabel('time [s]')
plt.yticks(size=11)
plt.xticks(size=11)
plt.show()
