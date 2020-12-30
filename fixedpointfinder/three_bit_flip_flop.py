import jax.numpy as np
from jax import value_and_grad, jit, random, vmap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Dense
from fixedpointfinder import rnn
from fixedpointfinder.plot_utils import visualize_flipflop

import time
import numpy as onp
import numpy.random as npr


def generate_flipflop_trials():
    '''Generates synthetic data (i.e., ground truth trials) for the
    FlipFlop task. See comments following FlipFlop class definition for a
    description of the input-output relationship in the task.

    Args:
        None.
    Returns:
        dict containing 'inputs' and 'outputs'.
            'inputs': [n_batch x n_time x n_bits] numpy array containing
            input pulses.
            'outputs': [n_batch x n_time x n_bits] numpy array specifying
            the correct behavior of the FlipFlop memory device.'''
    data_hps = {'n_batch': 128,
                 'n_time': 256,
                 'n_bits': 3,
                 'p_flip': 0.6}
    n_batch = data_hps['n_batch']
    n_time = data_hps['n_time']
    n_bits = data_hps['n_bits']
    p_flip = data_hps['p_flip']

    # Randomly generate unsigned input pulses
    unsigned_inputs = npr.binomial(
        1, p_flip, [n_batch, n_time, n_bits])

    # Ensure every trial is initialized with a pulse at time 0
    unsigned_inputs[:, 0, :] = 1

    # Generate random signs {-1, +1}
    random_signs = 2 * npr.binomial(
        1, 0.5, [n_batch, n_time, n_bits]) - 1

    # Apply random signs to input pulses
    inputs = onp.multiply(unsigned_inputs, random_signs)

    # Allocate output
    output = onp.zeros([n_batch, n_time, n_bits])

    # Update inputs (zero-out random start holds) & compute output
    for trial_idx in range(n_batch):
        for bit_idx in range(n_bits):
            input_ = onp.squeeze(inputs[trial_idx, :, bit_idx])
            t_flip = onp.where(input_ != 0)
            for flip_idx in range(onp.size(t_flip)):
                # Get the time of the next flip
                t_flip_i = t_flip[0][flip_idx]

                '''Set the output to the sign of the flip for the
                remainder of the trial. Future flips will overwrite future
                output'''
                output[trial_idx, t_flip_i:, bit_idx] = \
                    inputs[trial_idx, t_flip_i, bit_idx]

    return {'inputs': inputs, 'output': output}


def mse_loss(params, inputs, targets):
    """ Calculate the Mean Squared Error Prediction Loss. """
    preds = gru_rnn(params, inputs)
    return np.mean((preds - targets)**2)


@jit
def update(params, x, y, opt_state):
    """ Perform a forward pass, calculate the MSE & perform a SGD step. """
    loss, grads = value_and_grad(mse_loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, loss

num_batches = 128
batch_size = 128


def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx + 1) * batch_size)


key = random.PRNGKey(24)

n_hidden = 24


init_params = rnn.vanilla_params(key, 3, n_hidden, 3)

step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(init_params)

training_data = generate_flipflop_trials()


train_loss_log = []
start_time = time.time()
for epoch in range(100):
    for batch_idx in range(num_batches):

        # ids = batch_indices(batch_idx)
        x = np.expand_dims(training_data['inputs'][batch_idx, :, :], 0)
        y = np.expand_dims(training_data['output'][batch_idx, :, :], 0)

        opt_state = rnn.update_w_jit(batch_idx*epoch, opt_state, opt_update, get_params, x, y)
    batch_time = time.time() - start_time

    start_time = time.time()
    loss = rnn.loss_jit(get_params(opt_state), x, y)
    print("Epoch {} | T: {:0.2f} | MSE: {:0.10f} |".format(epoch, batch_time, loss))

x = np.expand_dims(training_data['inputs'][0, :, :], 0)
h_t, prediction = rnn.batch_rnn_run(get_params(opt_state), x)
prediction = onp.asarray(prediction)

visualize_flipflop(prediction, training_data)