import jax.numpy as np
from jax import grad, jit, random, vmap
from jax.experimental import optimizers


class Flipflopper(object):
    ''' Class for training an RNN to implement a 3-Bit Flip-Flop task as
    described in Sussillo, D., & Barak, O. (2012). Opening the Black Box:
    Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks.
    Neural Computation, 25(3), 626–649. https://doi.org/10.1162/NECO_a_00409

    Task:
        A set of three inputs submit transient pulses of either -1 or +1. The
        task of the model is to return said inputs until one of them flips.
        If the input pulse has the same value as the previous input in a given
        channel, the output must not change. Thus, the RNN has to memorize
        previous input pulses. The number of input channels is not limited
        in theory.

    Usage:
        The class Flipflopper can be used to build and train a model of type
        RNN on the 3-Bit Flip-Flop task. Furthermore, the class can make use
        of the class FixedPointFinder to analyze the trained model.

    Hyperparameters:
        rnn_type: Specifies architecture of type RNN. Must be one of
        ['vanilla','gru', 'lstm']. Will raise ValueError if
        specified otherwise. Default is 'vanilla'.

        n_hidden: Specifies the number of hidden units in RNN. Default
        is: 24.

    '''

    def __init__(self, rnn_type: str = 'vanilla', n_hidden: int = 24):

        self.hps = {'rnn_type': rnn_type,
                    'n_hidden': n_hidden,
                    'model_name': 'flipflopmodel',
                    'verbose': False}

        self.data_hps = {'n_batch': 128,
                         'n_time': 256,
                         'n_bits': 3,
                         'p_flip': 0.6}
        self.verbose = self.hps['verbose']

        key = random.PRNGKey(1)
        self.params, self.network = self._build_model(key, 3, n_hidden, 3)
        self.batch_network = vmap(self.network, in_axes=(None, 0, 0))

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)

        self.update_jit = jit(self.update)

    @staticmethod
    def _vanilla_params(key, n_input, n_hidden, n_output, scale=1e-3):

        keys = random.split(key, 5)

        UR = random.normal(next(keys), (n_hidden, n_input)) * scale
        WR = random.normal(next(keys), (n_hidden, n_hidden)) * scale
        bR = random.normal(next(keys), (n_hidden, )) * scale

        WO = random.normal(next(keys), (n_output, n_hidden)) * scale
        bO = random.normal(next(keys), (n_output, )) * scale

        return {'UR': UR,
                'WR': WR,
                'bR': bR,
                'WO': WO,
                'bO': bO}

    @staticmethod
    def vanilla_rnn(params, h, x):

        u = np.dot(params['UR'], x)
        r = np.dot(params['WR'], h) + params['bR']
        return np.tanh(r + u)

    @staticmethod
    def output_layer(params, h):

        return np.tanh(np.dot(params['WO'], h) + params['bO'])

    def _build_model(self, key, n_input, n_hidden, n_output):
        '''Builds model that can be used to train the 3-Bit Flip-Flop task.

        Args:
            None.

        Returns:
            None.'''
        params = self._vanilla_params(key, n_input, n_hidden, n_output)

        def network(params, x):

            hn = self.vanilla_rnn(params, self.h, x)
            self.h = hn
            return self.output_layer(params, hn), hn

        return params, network

    def loss(self, params, x, targets):

        outputs = self.batch_network(params, x)

        loss = np.mean((outputs - targets)**2)
        return loss

    def update(self, id, params, x, targets, opt_state):
        grads = grad(self.loss)(params, h, x, targets)
        opt_state = self.opt_update(id, grads, opt_state)
        return self.get_params(opt_state), opt_state

    def generate_flipflop_trials(self):
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
        data_hps = self.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        n_bits = data_hps['n_bits']
        p_flip = data_hps['p_flip']

        # Randomly generate unsigned input pulses
        unsigned_inputs = self.rng.binomial(
            1, p_flip, [n_batch, n_time, n_bits])

        # Ensure every trial is initialized with a pulse at time 0
        unsigned_inputs[:, 0, :] = 1

        # Generate random signs {-1, +1}
        random_signs = 2 * self.rng.binomial(
            1, 0.5, [n_batch, n_time, n_bits]) - 1

        # Apply random signs to input pulses
        inputs = np.multiply(unsigned_inputs, random_signs)

        # Allocate output
        output = np.zeros([n_batch, n_time, n_bits])

        # Update inputs (zero-out random start holds) & compute output
        for trial_idx in range(n_batch):
            for bit_idx in range(n_bits):
                input_ = np.squeeze(inputs[trial_idx, :, bit_idx])
                t_flip = np.where(input_ != 0)
                for flip_idx in range(np.size(t_flip)):
                    # Get the time of the next flip
                    t_flip_i = t_flip[0][flip_idx]

                    '''Set the output to the sign of the flip for the
                    remainder of the trial. Future flips will overwrite future
                    output'''
                    output[trial_idx, t_flip_i:, bit_idx] = \
                        inputs[trial_idx, t_flip_i, bit_idx]

        return {'inputs': inputs, 'output': output}

    def train(self, stim, epochs, save_model: bool = True):
        '''Function to train an RNN model This function will save the trained model afterwards.

        Args:
            stim: dict containing 'inputs' and 'output' as input and target data for training the model.

                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.
                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the FlipFlop memory device.

        Returns:
            None.'''

        return history
