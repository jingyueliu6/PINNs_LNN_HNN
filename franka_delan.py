import random
from random import shuffle

import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
import time
import optax
import numpy as np
import pickle

# package for plot
import matplotlib.pyplot as plt

import DeLaN_model_v3 as delan
from utils import ReplayMemory
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


print("Loading Data:")
'''
1. Data Loading and data preprocess
'''
with open(f"./data/franka_data_real_500Hz.jax", 'rb') as f:
    data_information = pickle.load(f)

time_step = data_information["time_step"]
states = data_information["states"]
inputs = data_information["input"]
targets = data_information["targets"]

assert states.shape[0] == targets.shape[0]
assert states.shape[0] == inputs.shape[0]

print(f"There are {states.shape[0]} pairs of data")


random.seed(42)
# shuffle
print("Shuffling Data ............")
shuffle_data = True
if shuffle_data:
    c = list(zip(states, inputs, targets))
    shuffle(c)
    states, inputs, targets = zip(*c)

states = np.array(states) #
inputs = np.array(inputs)
targets = np.array(targets)

assert states.shape[0] == inputs.shape[0]
assert states.shape[0] == targets.shape[0]

states = states[:35000, :]
inputs = inputs[:35000, :]
targets = targets[:35000, :]

print(f"Selecting {states.shape[0]} pairs of data")

'''
divide them into train and test set
'''
print(f"There are {states.shape[0]} data in this attempt.")
print("Set 70%  data as the training set and 30% as the test set")
div = states.shape[0] * 7 // 10
train_states, test_states = states[:div, :], states[div:, :]
train_targets, test_targets = targets[:div, :], targets[div:, :]
train_inputs, test_inputs = inputs[:div, :], inputs[div:, :]

train_q, train_dq = jnp.split(train_states, 2, axis=1)
train_q_next, train_dq_next = jnp.split(train_targets, 2, axis=1)

test_q, test_dq = jnp.split(test_states, 2, axis=1)
test_q_next, test_dq_next = jnp.split(test_targets, 2, axis=1)


'''
2. Set the model parameters/hyperparameters
'''

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

actuator_dof = train_inputs.shape[1]
config_dof = train_q.shape[1]
print(f"The system has {config_dof} degrees of freedom")
print(f"The system has {actuator_dof} actuation dimension")

save_model = True
hyper = {
    'n_dof': config_dof,
    'actuator_dof': actuator_dof,
    'n_width_l': 40,
    'n_depth_l': 3,
    'n_width_d': 20,
    'n_depth_d': 2,
    'n_minibatch': 1000,
    'diagonal_epsilon': 0.3,
    'diagonal_shift': 1.0,
    'activation1': 'softplus',
    'activation2': 'tanh',
    'activation3': 'sigmoid',
    'learning_rate': 5.2e-03,
    'weight_decay': 1.0e-04,
    'max_epoch': int(0.5 * 1e4),
}

# 1. Construct DeLaN:
t0 = time.perf_counter()


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


rng = jax.random.PRNGKey(42)

mem_dim = ((config_dof,), (config_dof,), (actuator_dof,), (config_dof,), (config_dof,))
mem = ReplayMemory(train_q.shape[0], hyper["n_minibatch"], mem_dim)
mem.add_samples([train_q, train_dq, train_inputs, train_q_next, train_dq_next])

rng_key, init_key = jax.random.split(rng)
# Initialize Parameters:
q, dq, tau, q_next, dq_next = [jnp.array(x) for x in next(iter(mem))]
l_params = lagrangian_fn.init(init_key, q[0], dq[0])
d_params = dissipative_fn.init(init_key, dq[0])


params = {"lagrangian": l_params, "dissipative": d_params}

lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply


forward_model = delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, n_dof=config_dof)

state0 = jnp.concatenate([q[0], dq[0]])
_ = forward_model(state0, tau[0])

t_build = time.perf_counter() - t0
print(f"DeLaN Build Time    = {t_build:.2f}s")


# 2. Generate and initialize the optimizer
t0 = time.perf_counter()

optimizer1 = optax.adamw(
    learning_rate=hyper['learning_rate'],
    weight_decay=hyper['weight_decay']
)

optimizer2 = optax.adamw(
    learning_rate=hyper['learning_rate'],
    weight_decay=hyper['weight_decay']
)


opt1 = optimizer1.init(params["lagrangian"])
opt2 = optimizer2.init(params["dissipative"])


# def loss_fn(params, q, qd, tau, q_next, qd_next, lagrangian,
#             dissipative_mat, input_mat, n_dof, time_step=None):
loss_fn = partial(
    delan.loss_fn,
    lagrangian=lagrangian,
    dissipative_mat=dissipative_mat,
    n_dof=config_dof,
    time_step=time_step)


def update_fn(params, opt1, opt2, q, dq, tau, q_next, dq_next):
    (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, dq, tau, q_next, dq_next)
    # print(grads)
    updates1, opt1 = optimizer1.update(grads["lagrangian"], opt1, params["lagrangian"])
    params["lagrangian"] = optax.apply_updates(params["lagrangian"], updates1)
    updates2, opt2 = optimizer2.update(grads["dissipative"], opt2, params["dissipative"])
    params["dissipative"] = optax.apply_updates(params["dissipative"], updates2)
    return params, opt1, opt2, logs


update_fn = jax.jit(update_fn)
_, _, _, logs = update_fn(params, opt1, opt2, train_q[:2], train_dq[:2], train_inputs[:2], train_q_next[:2], train_dq_next[:2])

t_build = time.perf_counter() - t0
print(f"Optimizer Build Time = {t_build:.2f}s")

'''
3. Start Training Loop:
'''
t0_start = time.perf_counter()

train_losses = {"forward_loss": [],
                "forward_var": []}

test_losses = {"forward_loss": [],
               "forward_var": []}


print("")
epoch_i = 0
step = 50
while epoch_i < hyper['max_epoch']:
    n_batches = 0
    logs = jax.tree_map(lambda x: x * 0.0, logs)

    for data_batch in mem:
        t0_batch = time.perf_counter()

        q, dq, tau, q_next, dq_next= [jnp.array(x) for x in data_batch]
        params, opt1, opt2, batch_logs = update_fn(params, opt1, opt2, q, dq, tau, q_next, dq_next)

        # Update logs:
        n_batches += 1
        logs = jax.tree_map(lambda x, y: x + y, logs, batch_logs)
        t_batch = time.perf_counter() - t0_batch

    # Update Epoch Loss & Computation Time:
    epoch_i += 1
    logs = jax.tree_map(lambda x: x / n_batches, logs)

    if epoch_i == 1 or np.mod(epoch_i, step) == 0:
        print("Epoch {0:05d}: ".format(epoch_i), end=" ")
        print("train: ", end=" ")
        print(f"Time = {time.perf_counter() - t0_start:05.1f}s", end=", ")
        print(f"For = {logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(logs['forward_var']):.2e}", end=";     ")
        train_losses["forward_loss"].append(logs['forward_mean'])
        train_losses["forward_var"].append(logs['forward_var'])

        # test loss computation:
        # (params, q, p, tau, q_next, p_next)
        test_loss, test_logs = loss_fn(params=params, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next)
        print("test: ", end=" ")
        print(f"For = {test_logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(test_logs['forward_var']):.2e}")
        test_losses["forward_loss"].append(test_logs['forward_mean'])
        test_losses['forward_var'].append(test_logs['forward_var'])


plt.style.use('seaborn-whitegrid')
palette = plt.get_cmap('Set1')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 5,
         }

iterations = list(range(0, hyper['max_epoch']+step, step))
f, ax = plt.subplots(1, 1)

# train forward loss
ax.plot(iterations, train_losses["forward_loss"], color='blue', label='train_forward_loss')
r1_forward = list(map(lambda x: x[0] - x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
r2_forward = list(map(lambda x: x[0] + x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
ax.fill_between(iterations, r1_forward, r2_forward, color='blue', alpha=0.2)


# train forward loss
ax.plot(iterations, test_losses["forward_loss"], color='g', label='test_forward_loss')
t1_forward = list(map(lambda x: x[0] - x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
t2_forward = list(map(lambda x: x[0] + x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
ax.fill_between(iterations, t1_forward, t2_forward, color='g', alpha=0.2)

ax.legend(loc='upper right', prop=font1)
ax.set_xlim(0, hyper['max_epoch']+1)
# ax.set_ylim(0, 2.5e-6)
ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('loss', fontsize=12)
plt.show()

if save_model:
    with open(f"./models/franka_real_delan.jax", "wb") as file:
        pickle.dump(
            {"epoch": epoch_i,
             "hyper": hyper,
             "params": params},
            file)