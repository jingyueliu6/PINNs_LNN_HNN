import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial



def rk4_step(f, x, y, t, h):
    # Runge-kutta integration
    k1 = h * f(x, y, t)
    k2 = h * f(x + k1/2, y, t + h/2)
    k3 = h * f(x + k2/2, y, t + h/2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

# def mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift):
#     n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix
#
#     # Calculate the indices of the diagonal elements of L:
#     idx_diag = np.arange(n_dof, dtype=int) + 1
#     idx_diag = (idx_diag * (idx_diag + 1) / 2 - 1).astype(int)
#     # [0 2 5]
#     '''
#     0 * *
#     1 2 *
#     3 4 5
#     '''
#
#     # Calculate the indices of the off-diagonal elements of L:
#     idx_tril = np.setdiff1d(np.arange(n_output), idx_diag)
#     # [1, 3, 4]
#
#     # Indexing for concatenation of l_diagonal and l_off_diagonal
#     cat_idx = np.hstack((idx_diag, idx_tril))
#     # [0 2 5 1 3 4]
#     idx = np.arange(cat_idx.size)[np.argsort(cat_idx)]
#     # [0 2 5 1 3 4]
#     # [0 1 2 3 4 5]
#     # [0 3 1 4 5 2]
#
#     # Compute Matrix Indices
#     mat_idx = np.tril_indices(n_dof)
#     # (array([0, 1, 1, 2, 2, 2]), array([0, 0, 1, 0, 1, 2]))
#     # the first is the row num, the second is the column num, of the lower triangle matrix
#     # mat_idx = jax.ops.index[..., mat_idx[0], mat_idx[1]]
#
#     # Compute Matrix Indices
#     net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
#                       activation=activation,
#                       name="mass_matrix")
#
#     l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)
#
#     # Ensure positive diagonal:
#     l_diagonal = jax.nn.softplus(l_diagonal + shift) + epsilon
#
#     vec_lower_triangular = jnp.concatenate((l_diagonal, l_off_diagonal), axis=-1)[..., idx]
#
#     triangular_mat = jnp.zeros((n_dof, n_dof))
#     triangular_mat = triangular_mat.at[mat_idx].set(vec_lower_triangular[:])
#     mass_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
#     return mass_mat
#
#
# def dissipative_matrix(q, n_dof, shape, activation):
#     assert n_dof > 0
#     n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix
#
#     # Calculate the indices of the diagonal elements of L:
#     idx_diag = np.arange(n_dof, dtype=int) + 1
#     idx_diag = (idx_diag * (idx_diag + 1) / 2 - 1).astype(int)
#
#     # Calculate the indices of the off-diagonal elements of L:
#     idx_tril = np.setdiff1d(np.arange(n_output), idx_diag)
#
#     # Indexing for concatenation of l_diagonal and l_off_diagonal
#     cat_idx = np.hstack((idx_diag, idx_tril))
#
#     idx = np.arange(cat_idx.size)[np.argsort(cat_idx)]
#
#     # Compute Matrix Indices
#     mat_idx = np.tril_indices(n_dof)
#
#     # the first is the row num, the second is the column num, of the lower triangle matrix
#     # mat_idx = jax.ops.index[..., mat_idx[0], mat_idx[1]]
#
#     # Compute Matrix Indices
#     net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
#                       activation=activation,
#                       name="dissipative_matrix")
#
#     # scaler to constraint the matrix
#     scaler = 0.4
#     l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)
#
#     l_diagonal = jax.nn.sigmoid(l_diagonal)
#
#     vec_lower_triangular = jnp.concatenate((l_diagonal, l_off_diagonal), axis=-1)[..., idx]
#
#     triangular_mat = jnp.zeros((n_dof, n_dof))
#
#     triangular_mat = triangular_mat.at[mat_idx].set(vec_lower_triangular[:])
#     dissipative_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
#     dissipative_mat *= scaler
#     return dissipative_mat


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
    # input_mat = jax.nn.tanh(net(q)).reshape(n_dof, actuator_dof)
    '''
    Adding the following line for the two segment which A elements are always between -1 t0 1
    '''
    # n_output = int(n_dof * actuator_dof / 2)
    # net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
    #                   activation=activation,
    #                   name="input_transform_matrix")
    # # l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)
    # elems = jax.nn.tanh(net(q))
    # seg1, seg2 = jnp.split(elems, [int(n_output/2), ], axis=-1)
    # seg1_mat = seg1.reshape(int(n_dof/2), int(n_dof/2))
    # seg2_mat = seg2.reshape(int(n_dof/2), int(n_dof/2))
    # left = jnp.row_stack((seg1_mat, jnp.zeros((int(n_dof/2), int(n_dof/2)))))
    # right = jnp.row_stack((jnp.zeros((int(n_dof / 2), int(n_dof / 2))), seg2_mat))
    # input_mat = jnp.hstack((left, right))
    return input_mat


def kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift):
    mass_mat = mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift)
    return 1./2. * jnp.dot(qd, jnp.dot(mass_mat, qd))


def potential_energy_fn(q, shape, activation):
    net = hk.nets.MLP(output_sizes=shape +(1, ),
                      activation=activation,
                      name="potential_energy")

    # Apply feature transform
    return net(q)


def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
    e_kin = kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift)
    e_pot = potential_energy_fn(q, shape, activation).squeeze()
    return e_kin - e_pot


def forward_model(params, key, lagrangian, dissipative_mat, input_mat, n_dof):
    def equation_of_motion(state, tau, t=None):
        # state should be a (n_dof * 3) np.array
        q, qd = jnp.split(state, 2)
        argnums = [2, 3]

        l_params = params["lagrangian"]

        # Compute Lagrangian and Jacobians:
        # def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
        lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
        L, (dLdq, dLdqd) = lagrangian_value_and_grad(l_params, key, q, qd)

        # Compute Hessian:
        lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
        (_, (d2L_dqddq, d2Ld2qd)) = lagrangian_hessian(l_params, key, q, qd)

        # Compute Dissipative term
        d_params = params["dissipative"]
        # def dissipative_matrix(qd, n_dof, shape, activation):
        dissipative = dissipative_mat(d_params, key, q)

        # for A(q) as a net
        i_params = params["input_transform"]
        input_transform = input_mat(i_params, key, q)

        # input_transform = input_mat(q)
        # # Compute the forward model:
        # # qdd_pred = jnp.linalg.inv(d2Ld2qd + 1.e-7 * jnp.eye(n_dof)) @ \
        # #            (input_transform @ tau - d2L_dqddq @ qd + dLdq - dissipative @ qd)

        qdd_pred = jnp.linalg.pinv(d2Ld2qd) @ \
                   (input_transform @ tau - d2L_dqddq @ qd + dLdq - dissipative @ qd)
        # input_mat = np.array([[0], [1]])
        # qdd_pred = jnp.linalg.inv(d2Ld2qd + 1.e-4 * jnp.eye(n_dof)) @ \
        #            (input_mat @ tau - d2L_dqddq @ qd + dLdq - dissipative @ qd)

        return jnp.concatenate([qd, qdd_pred])
    return equation_of_motion


def inverse_model(params, key, lagrangian, dissipative_mat, input_mat, n_dof):
    def equation_of_motion(state, qdd=None,  t=None):
        # state should be a (n_dof * 3) np.array
        q, qd = jnp.split(state, 2)
        argnums = [2, 3]

        l_params = params["lagrangian"]

        # Compute Lagrangian and Jacobians:
        # def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
        lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
        L, (dLdq, dLdqd) = lagrangian_value_and_grad(l_params, key, q, qd)

        # Compute Hessian:
        lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
        (_, (d2L_dqddq, d2Ld2qd)) = lagrangian_hessian(l_params, key, q, qd)

        # Compute Dissipative term
        d_params = params["dissipative"]
        # def dissipative_matrix(qd, n_dof, shape, activation):
        dissipative = dissipative_mat(d_params, key, q)

        i_params = params["input_transform"]
        input_transform = input_mat(i_params, key, q)

        # Compute the inverse model
        tau = jnp.linalg.inv(input_transform) @ (d2Ld2qd @ qdd + d2L_dqddq @ qd - dLdq + dissipative @ qd)
    return equation_of_motion

def loss_fn(params, q, qd, tau, q_next, qd_next, lagrangian, dissipative_mat, input_mat, n_dof, time_step=None):
    # vmap_dim = (0, 0)
    states = jnp.concatenate([q, qd], axis=1)
    targets = jnp.concatenate([q_next, qd_next], axis=1)

    # Forward error:
    f = jax.jit(forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, input_mat=input_mat, n_dof=n_dof))
    if time_step is not None:
        preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step), (0, 0))(states, tau)
        # preds = jax.vmap(normalize_dp)(preds)
        # preds => [q_next_pred, qd_next_pred]
    else:
        preds = jax.vmap(f, (0, 0))(states, tau)
        # preds => [qd_pred, qdd_pred]

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