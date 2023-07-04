import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial



def rk4_step(f, x, y, t, h):
    # one step of runge-kutta integration
    # print("x: ", x)
    # print("y: ", y)
    # print(t)
    # print(h)
    k1 = h * f(x, y, t)
    k2 = h * f(x + k1/2, y, t + h/2)
    k3 = h * f(x + k2/2, y, t + h/2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift):
    assert n_dof > 0
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Calculate the indices of the diagonal elements of L:
    idx_diag = np.arange(n_dof, dtype=int) + 1
    idx_diag = (idx_diag * (idx_diag + 1) / 2 - 1).astype(int)
    # [0 2 5]
    '''
    0 * *
    1 2 *
    3 4 5
    '''

    # Calculate the indices of the off-diagonal elements of L:
    idx_tril = np.setdiff1d(np.arange(n_output), idx_diag)
    # [1, 3, 4]

    # Indexing for concatenation of l_diagonal and l_off_diagonal
    cat_idx = np.hstack((idx_diag, idx_tril))
    # [0 2 5 1 3 4]
    idx = np.arange(cat_idx.size)[np.argsort(cat_idx)]
    # [0 2 5 1 3 4]
    # [0 1 2 3 4 5]
    # [0 3 1 4 5 2]

    # Compute Matrix Indices
    mat_idx = np.tril_indices(n_dof)
    # (array([0, 1, 1, 2, 2, 2]), array([0, 0, 1, 0, 1, 2]))
    # the first is the row num, the second is the column num, of the lower triangle matrix
    # mat_idx = jax.ops.index[..., mat_idx[0], mat_idx[1]]

    # Compute Matrix Indices
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="mass_matrix")

    # Apply feature transform:
    l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)

    # Ensure positive diagonal:
    l_diagonal = jax.nn.softplus(l_diagonal + shift) + epsilon

    vec_lower_triangular = jnp.concatenate((l_diagonal, l_off_diagonal), axis=-1)[..., idx]

    triangular_mat = jnp.zeros((n_dof, n_dof))
    # triangular_mat = jax.ops.index_update(triangular_mat, mat_idx, vec_lower_triangular[:])
    triangular_mat = triangular_mat.at[mat_idx].set(vec_lower_triangular[:])
    mass_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    return mass_mat


# For HNN the predicted mass matrix inverse is composed of the LL^{T} + eps I.
inv_mass_matrix_fn = mass_matrix_fn


def dissipative_matrix(p, n_dof, shape, activation):
    assert n_dof > 0
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Calculate the indices of the diagonal elements of L:
    idx_diag = np.arange(n_dof, dtype=int) + 1
    idx_diag = (idx_diag * (idx_diag + 1) / 2 - 1).astype(int)

    # Calculate the indices of the off-diagonal elements of L:
    idx_tril = np.setdiff1d(np.arange(n_output), idx_diag)

    # Indexing for concatenation of l_diagonal and l_off_diagonal
    cat_idx = np.hstack((idx_diag, idx_tril))

    idx = np.arange(cat_idx.size)[np.argsort(cat_idx)]

    # Compute Matrix Indices
    mat_idx = np.tril_indices(n_dof)

    # the first is the row num, the second is the column num, of the lower triangle matrix
    # mat_idx = jax.ops.index[..., mat_idx[0], mat_idx[1]]

    # Compute Matrix Indices
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="dissipative_matrix")

    # Apply feature transform:
    l_diagonal, l_off_diagonal = jnp.split(net(p), [n_dof, ], axis=-1)

    l_diagonal = jax.nn.softplus(l_diagonal)

    vec_lower_triangular = jnp.concatenate((l_diagonal, l_off_diagonal), axis=-1)[..., idx]

    triangular_mat = jnp.zeros((n_dof, n_dof))

    triangular_mat = triangular_mat.at[mat_idx].set(vec_lower_triangular[:])
    dissipative_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    return dissipative_mat


def input_transform_matrix(q, n_dof, actuator_dof, shape, activation):
    n_output = n_dof * actuator_dof
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="input_transform_matrix")
    input_mat = net(q).reshape(n_dof, actuator_dof)
    return input_mat



def potential_energy_fn(q, shape, activation):
    net = hk.nets.MLP(output_sizes=shape +(1, ),
                      activation=activation,
                      name="potential_energy")

    # Apply feature transform
    return net(q)


def kinetic_energy_fn(q, p, n_dof, shape, activation, epsilon, shift):
    inv_mass_mat = inv_mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift)
    return 1. / 2. * jnp.dot(p, jnp.dot(inv_mass_mat, p))


def structured_hamiltonian_fn(q, p, n_dof, shape, activation, epsilon, shift):
    e_kin = kinetic_energy_fn(q, p, n_dof, shape, activation, epsilon, shift)
    e_pot = potential_energy_fn(q, shape, activation).squeeze()
    return e_kin + e_pot


def blackbox_hamiltonian_fn(q, p, n_dof, shape, activation, epsilon, shift):
    del epsilon, n_dof
    net = hk.nets.MLP(output_sizes= shape + (1,),
                      activation=activation,
                      name="hamiltonian")

    z = jnp.concatenate([jnp.cos(q), jnp.sin(q)], axis=-1)
    state = jnp.concatenate([z, p], axis=-1)
    return net(state).squeeze()


def forward_model(params, key, hamiltonian, dissipative_mat, input_mat):
    def hamiltons_equation(state, tau, t=None):
        q, p = jnp.split(state, 2)
        argnums = [2, 3]
        # print(state)
        # print(tau)

        h_params = params["hamilton"]
        # Compute Hamiltonian and Jacobians:
        hamiltonian_value_and_grad = jax.value_and_grad(hamiltonian, argnums=argnums)
        H, (dHdq, dHdp) = hamiltonian_value_and_grad(h_params, key, q, p)

        d_params = params["dissipative"]
        dissipative = dissipative_mat(d_params, key, q)
        #
        # for A(q) as a net
        i_params = params["input_transform"]
        input_transform = input_mat(i_params, key, q)

        tau_prime = input_transform @ tau

        # ## for A(q) was given
        # tau_prime = input_mat @ tau

        pd_pred = tau_prime - dHdq - dissipative @ dHdp
        qd_pred = dHdp
        return jnp.concatenate([qd_pred, pd_pred])
    return hamiltons_equation


def loss_fn(params, q, p, tau, q_next, p_next, hamiltonian, dissipative_mat, input_mat, time_step=None):
    vmap_dim = (0, 0)
    states = jnp.concatenate([q, p], axis=1)
    targets = jnp.concatenate([q_next, p_next], axis=1)
    # print(states[0, :])
    # print(tau[0, :])
    # Forward error:
    f = jax.jit(forward_model(params=params, key=None, hamiltonian=hamiltonian, dissipative_mat=dissipative_mat, input_mat=input_mat))
    if time_step is not None:
        preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step), vmap_dim)(states, tau)
    else:
        preds = jax.vmap(f, vmap_dim)(states, tau)

    # alpha = 10
    # n_dof = q.shape[1]
    # position_error = jnp.sum((q_next - preds[:, :n_dof])**4, axis=-1)
    # momentum_error = jnp.sum((p_next - preds[:, :n_dof:])**2, axis=-1)
    forward_error = jnp.sum((targets - preds)**2, axis=-1)
    mean_forward_error = jnp.mean(forward_error)
    var_forward_error = jnp.var(forward_error)
    # mean_forward_error = jnp.mean(position_error) + jnp.mean(momentum_error)
    # var_forward_error = jnp.mean(momentum_error) + jnp.mean(position_error)

    # Compute Loss
    loss =  mean_forward_error

    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
    }
    return loss, logs