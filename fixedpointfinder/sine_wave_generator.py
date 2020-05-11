import tensorflow as tf
import numpy as np
import numpy.random as npr
from utilities.model_utils import build_sub_model_to
from tensorflow.keras.models import load_model
import os
from tensorflow.python.keras.utils import tf_utils
from mpl_toolkits.mplot3d import Axes3D


class SineWaveGenerator(object):
    ''' Class for training an RNN to implement a SineWaveGenerator task as
        described in Sussillo, D., & Barak, O. (2012). Opening the Black Box:
        Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks.
        Neural Computation, 25(3), 626–649. https://doi.org/10.1162/NECO_a_00409'''

    def __init__(self, rnn_type: str = 'vanilla', n_hidden: int = 24):

        self.rnn_type = rnn_type
        self.n_hidden = n_hidden
        self.name = 'sine_generator'
        self.verbose = True

        self.data_hps = {'n_batch': 100,
                         'n_time': 5,
                         'dt': 0.01}
        self.n_time = 150

        self.frequency_fun = lambda x: 0.04 * x + 0.01

        self.rng = npr.RandomState(125)
        self.n_frequencies = 1
        self.frequencies = np.linspace(0.4, 0.6, self.n_frequencies)

        self.model = self._build_model()

    def _build_model(self):

        '''Builds model that can be used to train the 3-Bit Flip-Flop task.

        Args:
            None.

        Returns:
            None.'''
        n_hidden = self.n_hidden
        name = self.name

        n_samples, n_batch, n_time = self.n_frequencies, self.data_hps['n_batch'], self.n_time

        inputs = tf.keras.Input(shape=(n_time, self.n_frequencies), batch_size=n_batch, name='input')

        if self.rnn_type == 'vanilla':
            x = tf.keras.layers.SimpleRNN(n_hidden, name=self.rnn_type, return_sequences=True)(inputs)
        elif self.rnn_type == 'gru':
            x = tf.keras.layers.GRU(n_hidden, name=self.rnn_type, return_sequences=True)(inputs)
        elif self.rnn_type == 'lstm':
            x, state_h, state_c = tf.keras.layers.LSTM(n_hidden, name=self.rnn_type, return_sequences=True,
                                                       stateful=True, return_state=True,
                                                       implementation=1)(inputs)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.rnn_type)

        x = tf.keras.layers.Dense(self.n_frequencies)(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        # weights = model.get_layer(self.rnn_type).get_weights()

        if self.verbose:
            model.summary()

        return model

    def generate_stimulus(self):

        stimulus = {'input': np.zeros((self.data_hps['n_batch'], self.n_time, self.n_frequencies)),
                    'output': np.zeros((self.data_hps['n_batch'], self.n_time, self.n_frequencies))
                    }

        for i in range(self.data_hps['n_batch']):
            for k in range(self.n_frequencies):
                x = self.frequencies[k]
                input = self.frequency_fun(np.repeat(x, self.n_time))#.reshape(self.n_time, 1))
                stimulus['input'][i, :, k] = input
                output = np.sin(2*np.pi*self.frequency_fun(x)*np.linspace(1, self.n_time, self.n_time))
                stimulus['output'][i, :, k] = output#.reshape(self.n_time, 1)

        return stimulus

    def train(self, stimulus, n_epochs: int = 1000):

        self.model.compile(optimizer="adam", loss="mse",
                           metrics=['mse'])
        history = self.model.fit(tf.convert_to_tensor(stimulus['input'], dtype=tf.float32),
                                 tf.convert_to_tensor(stimulus['output'], dtype=tf.float32), epochs=n_epochs)
        return history

    def get_activations(self, stim):
        sub_model = build_sub_model_to(self.model, [self.rnn_type])
        activation = sub_model.predict(tf.convert_to_tensor(stim['input'], dtype=tf.float32))

        return activation


if __name__ == "__main__":

    rnn_type = 'vanilla'
    n_hidden = 24

    sinwaver = SineWaveGenerator(rnn_type=rnn_type, n_hidden=n_hidden)

    stimulus = sinwaver.generate_stimulus()
    history = sinwaver.train(stimulus, 400)
    import matplotlib.pyplot as plt

    prediction = sinwaver.model.predict(tf.convert_to_tensor(stimulus['input'], dtype=tf.float32))

    plt.plot(prediction[0, :, :], label='prediction')
    plt.plot(stimulus['output'][0, :, :], label='data')
    plt.legend(loc='upper right')
    plt.show()

    #plt.plot(stimulus['input'][0, :, :])
    #plt.show()

    import sklearn.decomposition as skld

    activation = sinwaver.get_activations(stimulus)
    pca = skld.PCA(3)
    x_pca = pca.fit_transform(activation[0, :, :])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n_points = 150
    ax.plot(x_pca[:n_points, 0], x_pca[:n_points, 1], x_pca[:n_points, 2],
            linewidth=0.7)
