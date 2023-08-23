import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial



def rk4_step(f, x, y, t, h):
    # one step of runge-kutta integration
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


def forward_model(params, key, lagrangian, dissipative_mat, n_dof):
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

        # Compute the forward model:
        qdd_pred = jnp.linalg.inv(d2Ld2qd) @ \
                   (tau - d2L_dqddq @ qd + dLdq - dissipative @ qd)
        # input_mat = np.array([[0], [1]])
        # qdd_pred = jnp.linalg.inv(d2Ld2qd + 1.e-4 * jnp.eye(n_dof)) @ \
        #            (input_mat @ tau - d2L_dqddq @ qd + dLdq - dissipative @ qd)

        return jnp.concatenate([qd, qdd_pred])
    return equation_of_motion


def inverse_model(params, key, lagrangian, dissipative_mat, n_dof):
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

        # Compute the inverse model
        if qdd is None:
            tau = d2L_dqddq @ qd - dLdq + dissipative @ qd
        else:
            tau = d2Ld2qd @ qdd + d2L_dqddq @ qd - dLdq + dissipative @ qd
        return tau
    return equation_of_motion


def loss_fn(params, q, qd, tau, q_next, qd_next, lagrangian, dissipative_mat, n_dof, time_step=None):
    vmap_dim = (0, 0)
    states = jnp.concatenate([q, qd], axis=1)
    targets = jnp.concatenate([q_next, qd_next], axis=1)

    # Forward error:
    f = jax.jit(forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, n_dof=n_dof))
    if time_step is not None:
        preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step), vmap_dim)(states, tau)
        # preds = jax.vmap(normalize_dp)(preds)
        # preds => [q_next_pred, qd_next_pred]
    else:
        preds = jax.vmap(f, vmap_dim)(states, tau)
        # preds => [qd_pred, qdd_pred]
    # f_inv = jax.jit(inverse_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, n_dof=n_dof))
    # preds_inv = jax.vmap(f_inv)(states)

    forward_error = jnp.sum((targets - preds)**2, axis=-1)
    mean_forward_error = jnp.mean(forward_error)
    var_forward_error = jnp.var(forward_error)

    # inverse_error = jnp.sum((tau - preds_inv)**2, axis=-1)
    # mean_inverse_error = jnp.mean(inverse_error)
    # var_inverse_error = jnp.var(inverse_error)
    # Compute Loss
    # loss = 100 * mean_forward_error + 0.1 * mean_inverse_error
    loss = mean_forward_error
    # print(loss.shape)
    # print("loss: ", loss)
    # logs = {
    #     'loss': loss,
    #     'forward_mean': mean_forward_error,
    #     'forward_var': var_forward_error,
    # }
    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
    }
    return loss, logs
