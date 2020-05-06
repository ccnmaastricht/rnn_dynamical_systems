import sys
sys.path.append("/Users/Raphael/dexterous-robot-hand/rnn_dynamical_systems")
from rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder
from rnn_dynamical_systems.fixedpointfinder.three_bit_flip_flop import Flipflopper
from rnn_dynamical_systems.fixedpointfinder.plot_utils import plot_fixed_points, plot_velocities, visualize_flipflop
import autograd.numpy as np
import os
import tensorflow as tf
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    ############################################################
    # Create and train recurrent model on 3-Bit FlipFop task
    ############################################################
    # specify architecture e.g. 'vanilla' and number of hidden units
    rnn_type = 'gru'
    n_hidden = 24

    # initialize Flipflopper class
    flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
    # generate trials
    stim = flopper.generate_flipflop_trials()
    # train the model
    # flopper.train(stim, 2000, save_model=True)

    # visualize a single batch after training
    # prediction = flopper.model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))
    # visualize_flipflop(prediction, stim)

    # if a trained model has been saved, it may also be loaded
    flopper.load_model()
    ############################################################
    # Initialize fpf and find fixed points
    ############################################################
    # get weights and activations of trained model
    weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
    activations = flopper.get_activations(stim)
    # initialize adam fpf
    fpf = Adamfixedpointfinder(weights, rnn_type,
                               q_threshold=1e-12,
                               epsilon=0.01,
                               alr_decayr=0.0001,
                               max_iters=5000)
    # sample states, i.e. a number of ICs
    states = fpf.sample_states(activations, 1000, 0.5)
    # vel = fpf.compute_velocities(np.hstack(activations[1:]), np.zeros((32768, 3)))
    # generate corresponding input as zeros for flip flop task
    # please keep in mind that the input does not need to be zero for all tasks
    inputs = np.zeros((states.shape[0], 3))
    # find fixed points
    fps = fpf.find_fixed_points(states, inputs)
    # plot fixed points and state trajectories in 3D
    plot_fixed_points(activations, fps, 3000, 1)
