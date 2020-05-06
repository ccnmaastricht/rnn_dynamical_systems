import sys
sys.path.append("/Users/Raphael/dexterous-robot-hand/rnn_dynamical_systems")
from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from analysis.rnn_dynamical_systems.fixedpointfinder.three_bit_flip_flop import Flipflopper
from analysis.rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points, visualize_flipflop
import autograd.numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    ############################################################
    # Create and train recurrent model on 3-Bit FlipFop task
    ############################################################
    # specify architecture e.g. 'vanilla' and number of hidden units
    rnn_type = 'vanilla'
    n_hidden = 24

    # initialize Flipflopper class
    flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
    # generate trials
    stim = flopper.generate_flipflop_trials()
    # train the model
    # flopper.train(stim, 4000, save_model=True)

    # if a trained model has been saved, it may also be loaded
    flopper.load_model()

    # visualize a single batch after training
    prediction = flopper.model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))
    visualize_flipflop(prediction, stim)
    ############################################################
    # Initialize fpf and find fixed points
    ############################################################
    # get weights and activations of trained model
    weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
    activations = flopper.get_activations(stim)
    # initialize adam fpf
    fpf = Adamfixedpointfinder(weights, rnn_type,
                               q_threshold=1e-10,
                               epsilon=0.01,
                               alr_decayr=0.0001,
                               max_iters=7000)
    # sample states, i.e. a number of ICs
    states = fpf.sample_states(activations, 1000, 0.2)
    inputs = np.zeros((states.shape[0], 3))
    # find fixed points
    fps = fpf.find_fixed_points(states, inputs)
    # plot fixed points and state trajectories in 3D
    plot_fixed_points(activations, fps, 2000, 2.5)
    plt.show()
