# Fixedpointfinder

[![fpfoptim.gif](https://i.postimg.cc/63W65W9Q/fpfoptim.gif)](https://postimg.cc/1gd1vZVx)
---
The fixedpointfinder may be used to find fixed points in recurrent neural networks.
Concepts were derived from:

* Sussillo, D., & Barak, O. (2012). Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks. Neural Computation, 25(3), 626–649. https://doi.org/10.1162/NECO_a_00409
* Golub and Sussillo (2018), "FixedPointFinder: A Tensorflow toolbox for identifying and characterizing fixed points in recurrent neural networks," Journal of Open Source Software, 3(31), 1003, https://doi.org/10.21105/joss.01003

# Content
Fixedpointfinder comes in mutliple child classes. Those include:
  * `Adamfixedpointfinder`, a version employing Adam for minimization
  * `Scipyfixedpointfinder`, a version employing minimize from scipy's optimization package
  * `Tffixedpointfinder`, a `tensorflow` version
  * `RecordingFixedpointfinder`, a version employing Adam with functionality to record the optimization process
  
`Joint` optimization from `Adamfixedpointfinder` converges fastest empirically.
# Usage

1. Begin by building and training an RNN. Fixedpointfinder works with any framework. It simply requires 
parameters of the recurrent layer, the `rnn_type` and a set of hyperparameters.

2. Build an instance of `Fixedpointfinder` :  
`fpf = Adamfixedpointfinder(weights, rnn_type, **hyperparameters)`  
using the parameters of the recurrent layer, rnn_type as one of `vanilla`, `gru` and hyperparameters for optimization

3. Sample states or states and inputs to recurrent layer from a set of activations recorded during a trial of the task of interest.

4.  Find fixed points: `fps = fpf.find_fixed_points(states, inputs)`  
Note that inputs can be zero, if only the long term behavior of the system is of interest. This may be for instance if the 
input is made up of pulses.   
This function will return a fixedpointobject containing the fixedpoint location, initial state from which it was derived, 
jacobian matrix as linearization around the fixed point and function evaluation at the fixed point.

5. From `plot_utils` `plot_fixed_points` may now be used to visualize the trajectories together with found fixed points 
analyzed stability and a 3D dimensionality reduced plot. 

## Flip Flop

[![tbfflearning.gif](https://i.postimg.cc/0j8sJNNH/tbfflearning.gif)](https://postimg.cc/ZBDgG4pr)
---

The example file [fixedpoint_example](fixedpoint_example.py) deals with a three bit flip flop task.

* Sussillo, D., & Barak, O. (2012). Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks. Neural Computation, 25(3), 626–649. https://doi.org/10.1162/NECO_a_00409

The class to train the flip flop task using an rnn can be found in [three_bit_flip_flop](three_bit_flip_flop.py)

