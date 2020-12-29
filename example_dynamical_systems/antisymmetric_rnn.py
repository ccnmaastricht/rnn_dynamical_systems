from tensorflow.keras.layers import SimpleRNN, SimpleRNNCell
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K

from fixedpointfinder.three_bit_flip_flop import Flipflopper
from fixedpointfinder.plot_utils import visualize_flipflop


class AntisymmetricRNNCell(SimpleRNNCell):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AntisymmetricRNNCell, self).__init__(output_dim, **kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.gamma_I = tf.eye(self.units, self.units) * 0.2
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = h + K.dot(prev_output, (self.recurrent_kernel - tf.transpose(self.recurrent_kernel, perm=[1, 0]) - self.gamma_I))
        if self.activation is not None:
            output = self.activation(output)

        return output, [output]


class AntisymmetricRNN(SimpleRNN):

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='random_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = AntisymmetricRNNCell(units,
                                    activation=activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    recurrent_constraint=recurrent_constraint,
                                    bias_constraint=bias_constraint,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout,
                                    dtype=kwargs.get('dtype'))

        super(SimpleRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)


class AntisymmetricFlipFlopper(Flipflopper):

    def __init__(self,  rnn_type: str = 'vanilla', n_hidden: int = 24):

        super(AntisymmetricFlipFlopper, self).__init__(rnn_type, n_hidden)

    def _build_model(self):
        '''Builds model that can be used to train the 3-Bit Flip-Flop task.

        Args:
            None.

        Returns:
            None.'''

        n_hidden = self.hps['n_hidden']
        name = self.hps['model_name']

        n_time, n_batch, n_bits = self.data_hps['n_time'], self.data_hps['n_batch'], self.data_hps['n_bits']

        inputs = tf.keras.Input(shape=(n_time, n_bits), batch_size=n_batch, name='input')

        x = AntisymmetricRNN(n_hidden, name=self.hps['rnn_type'], return_sequences=True)(inputs)
        x = tf.keras.layers.Dense(3)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        weights = model.get_layer(self.hps['rnn_type']).get_weights()

        return model, weights


if __name__ == "__main__":
    n_hidden = 45
    rnn_type = "antisymmetric"

    # initialize Flipflopper class
    flopper = AntisymmetricFlipFlopper(rnn_type=rnn_type, n_hidden=n_hidden)
    # generate trials
    stim = flopper.generate_flipflop_trials()
    # train the model
    flopper.train(stim, 2000, save_model=True)

    prediction = flopper.model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))
    visualize_flipflop(prediction, stim)

