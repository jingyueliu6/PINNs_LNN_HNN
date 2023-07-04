import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial


def forward_model_fn(input, n_dof, shape, activation):
    net = hk.nets.MLP(output_sizes=shape + (n_dof * 2, ),
                      activation=activation,
                      name="black_box")

    # Apply feature transform
    return net(input) # the output is the acceleration


def forward_loss_fn(params, q, qd, tau, q_next, qd_next, forward_model):
    input = jnp.concatenate([q, qd, tau], axis=1)
    # print(input.shape)
    targets = jnp.concatenate([q_next, qd_next], axis=1)
    # state_transform(params, rng, input0)
    preds = forward_model(params, None, input)

    forward_error = jnp.sum((targets - preds)**2, axis=-1)
    mean_forward_error = jnp.mean(forward_error)
    var_forward_error = jnp.var(forward_error)
    loss = mean_forward_error

    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
    }
    return loss, logs