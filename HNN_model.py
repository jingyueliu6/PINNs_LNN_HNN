import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial



def rk4_step(f, x, y, t, h):
    k1 = h * f(x, y, t)
    k2 = h * f(x + k1/2, y, t + h/2)
    k3 = h * f(x + k2/2, y, t + h/2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift):
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Compute Matrix Indices
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="mass_matrix")

    l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)

    # Ensure positive diagonal:
    l_diagonal = jax.nn.softplus(l_diagonal + shift) + epsilon

    triangular_mat = jnp.zeros((n_dof, n_dof))
    diagonal_index = np.diag_indices(n_dof)
    tril_index = np.tril_indices(n_dof, -1)
    triangular_mat = triangular_mat.at[diagonal_index].set(l_diagonal[:])
    triangular_mat = triangular_mat.at[tril_index].set(l_off_diagonal[:])

    mass_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    return mass_mat


# For HNN the predicted mass matrix inverse is composed of the LL^{T} + eps I.
inv_mass_matrix_fn = mass_matrix_fn


def dissipative_matrix(q, n_dof, shape, activation):
    assert n_dof > 0
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Compute Matrix Indices
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="dissipative_matrix")

    # scaler to constraint the matrix
    scaler = 0.4
    l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)

    l_diagonal = jax.nn.sigmoid(l_diagonal)

    triangular_mat = jnp.zeros((n_dof, n_dof))
    diagonal_index = np.diag_indices(n_dof)
    tril_index = np.tril_indices(n_dof, -1)
    triangular_mat = triangular_mat.at[diagonal_index].set(l_diagonal[:])
    triangular_mat = triangular_mat.at[tril_index].set(l_off_diagonal[:])

    dissipative_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    dissipative_mat *= scaler
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

        pd_pred = tau_prime - dHdq - dissipative @ dHdp
        qd_pred = dHdp
        return jnp.concatenate([qd_pred, pd_pred])
    return hamiltons_equation


def loss_fn(params, q, p, tau, q_next, p_next, hamiltonian, dissipative_mat, input_mat, time_step=None):
    vmap_dim = (0, 0)
    states = jnp.concatenate([q, p], axis=1)
    targets = jnp.concatenate([q_next, p_next], axis=1)

    f = jax.jit(forward_model(params=params, key=None, hamiltonian=hamiltonian, dissipative_mat=dissipative_mat, input_mat=input_mat))
    if time_step is not None:
        preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step), vmap_dim)(states, tau)
    else:
        preds = jax.vmap(f, vmap_dim)(states, tau)


    forward_error = jnp.sum((targets - preds)**2, axis=-1)
    mean_forward_error = jnp.mean(forward_error)
    var_forward_error = jnp.var(forward_error)

    # Compute Loss
    loss =  mean_forward_error

    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
    }
    return loss, logs