from jax import random, vmap, grad, jit
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal

from jax import lax
import jax.numpy as np

from functools import partial


def GRU(out_dim, W_init=glorot_normal(), b_init=normal()):
    def init_fun(rng, input_shape):
        """ Initialize the GRU layer for stax """
        hidden = b_init(rng, (input_shape[0], out_dim))

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)
        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], input_shape[1], out_dim)
        return (output_shape,
            (hidden,
             (update_W, update_U, update_b),
             (reset_W, reset_U, reset_b),
             (out_W, out_U, out_b),),)

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = params[0]

        def apply_fun_scan(params, hidden, inp):
            """ Perform single step update of the network """
            _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
                out_W, out_U, out_b) = params

            update_gate = sigmoid(np.dot(inp, update_W) +
                                  np.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(np.dot(inp, reset_W) +
                                 np.dot(hidden, reset_U) + reset_b)
            output_gate = np.tanh(np.dot(inp, out_W)
                                  + np.dot(np.multiply(reset_gate, hidden), out_U)
                                  + out_b)
            output = np.multiply(update_gate, hidden) + np.multiply(1-update_gate, output_gate)
            hidden = output
            return hidden, hidden

        # Move the time dimension to position 0
        # inputs = np.moveaxis(inputs, 1, 0)
        f = partial(apply_fun_scan, params)
        _, h_new = lax.scan(f, h, inputs)
        return h_new

    return init_fun, apply_fun


def vanilla_params(key, n_input, n_hidden, n_output, scale=1e-3):

    keys = random.split(key, 6)
    keys = (k for k in keys)

    UR = random.normal(next(keys), (n_hidden, n_input)) * scale
    WR = random.normal(next(keys), (n_hidden, n_hidden)) * scale
    bR = random.normal(next(keys), (n_hidden, )) * scale

    # output, though technically not part of layer
    WO = random.normal(next(keys), (n_output, n_hidden)) * scale
    bO = random.normal(next(keys), (n_output, )) * 0
    return {'h0': random.normal(next(keys), (n_hidden, )) * 1e-4,
            'UR': UR,
            'WR': WR,
            'bR': bR,
            'WO': WO,
            'bO': bO}


def vanilla_rnn(params, h, x):

    u = np.dot(params['UR'], x)
    r = np.dot(params['WR'], h)
    return np.tanh(r + u + params['bR'])


def vanilla_scan(params, h, x):

    h = vanilla_rnn(params, h, x)
    return h, h


def output_layer(params, x):

    return np.dot(params['WO'], x) + params['bO']


batch_output_layer = vmap(output_layer, in_axes=(None, 0))


def vanilla_run_with_h0(params, x, h0):

    f = partial(vanilla_scan, params)

    _, h_t = lax.scan(f, h0, x)
    o = batch_output_layer(params, h_t)
    return h_t, o


def vanilla_run(params, x):

    return vanilla_run_with_h0(params, x, params['h0'])


batch_rnn_run = vmap(vanilla_run, in_axes=(None, 0))
batch_rnn_run_with_h0 = vmap(vanilla_run_with_h0, in_axes=(None, 0, 0))


def loss(params, x, targets):

    _, output = batch_rnn_run(params, x)

    loss = np.mean((output - targets)**2)
    return loss


def update_w(i, opt_state, opt_update, get_params,
             x, targets):

    params = get_params(opt_state)

    grads = grad(loss)(params, x, targets)
    return opt_update(i, grads, opt_state)


loss_jit = jit(loss)
update_w_jit = jit(update_w, static_argnums=(2, 3))
