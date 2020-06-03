import sklearn.decomposition as skld
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_velocities(activations, velocities, n_points: int = 5000):
    """This functions creates a 3D line plot of network activities
    colored by 'kinetic energy' i.e. rate of change in the dynamic system at
    any given position.

    Args:
        activations: Network activations. Must have dimensions [n_time, n_units].

        velocities: Vector of 'velocities' computed by feeding activations in q(x).
        Must have dimensions [n_time].

        n_points (optional): Number of points to be used for line plot. Recommended to choose values
         between 2000 and 5000. Default: 5000.

     Returns:
         None."""
    activations = np.vstack(activations)
    pca = skld.PCA(3)
    pca.fit(activations)
    X_pca = pca.transform(activations)

    mlab.plot3d(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
                velocities[:n_points], line_width=0.2)
    mlab.colorbar(orientation='vertical')
    mlab.show()


def plot_fixed_points(activations, fps, n_points, scale):
    """Plot a set of fixedpoints together with activations of a recurrent layer.

    Args:
        activations: numpy array containing activations of a recurrent layer in
        which dynamics have been analyzed
        fps: fixedpointobject containing a set of detected fixed points.
        n_points: Integer specifying how many datapoints of the activations
        dataset shall be plotted.
        scale: float specifying by how much the unstable modes shall be scaled for plotting.

    Returns:
        Plot of classified fixedpoints together with recurrent activations"""
    def extract_fixed_point_locations(fps):
        """Processing of minimisation results for pca. The function takes one fixedpoint object at a time and
        puts all coordinates in single array."""
        fixed_point_location = [fp['x'] for fp in fps]

        fixed_point_locations = np.vstack(fixed_point_location)

        return fixed_point_locations

    def classify_fixedpoints(fps, scale):
        """Function to classify fixed points. Methodology is based on
        'Nonlinear Dynamics and Chaos, Strogatz 2015'.

        Args:
            fps: Fixedpointobject containing a set of fixedpoints.
            scale: Float by which the unstable modes shall be scaled for plotting.

        Returns:
            fps: Fixedpointobject that contains 'fp_stability', i.e. information about
            the stability of the fixedpoint
            x_directions: list of matrices containing vectors of unstable modes"""

        x_directions = []
        scale = scale
        for fp in fps:

            trace = np.matrix.trace(fp['jac'])
            det = np.linalg.det(fp['jac'])
            if det > 0 and trace == 0:
                print('center has been found.')
            elif trace**2 - 4 * det == 0:
                print("star node has been found.")
            elif trace**2 - 4 * det < 0:
                print("spiral has been found")
            e_val, e_vecs = np.linalg.eig(fp['jac'])
            ids = np.argwhere(np.real(e_val) > 0)
            countgreaterzero = np.sum(e_val > 0)
            if countgreaterzero == 0:
                print('stable fixed point was found.')
                fp['fp_stability'] = 'stable fixed point'
            elif countgreaterzero > 0:
                print('saddle point was found.')
                fp['fp_stability'] = 'saddle point'
                for id in ids:
                    x_plus = fp['x'] + scale * e_val[id] * np.real(e_vecs[:, id].transpose())
                    x_minus = fp['x'] - scale * e_val[id] * np.real(e_vecs[:, id].transpose())
                    x_direction = np.vstack((x_plus, fp['x'], x_minus))
                    x_directions.append(np.real(x_direction))

        return fps, x_directions

    fps, x_directions = classify_fixedpoints(fps, scale)

    fixedpoints = extract_fixed_point_locations(fps)
    if len(activations.shape) == 3:
        activations = np.vstack(activations)

    pca = skld.PCA(3)
    pca.fit(activations)
    X_pca = pca.transform(activations)
    new_pca = pca.transform(fixedpoints)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
            linewidth=0.7)
    for i in range(len(new_pca)):
        if fps[i]['fp_stability'] == 'stable fixed point':
            ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                       marker='.', s=30, c='k')
        elif fps[i]['fp_stability'] == 'saddle point':
            ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                       marker='.', s=30, c='r')
            for p in range(len(x_directions)):
                direction_matrix = pca.transform(x_directions[p])
                ax.plot(direction_matrix[:, 0], direction_matrix[:, 1], direction_matrix[:, 2],
                        c='r', linewidth=0.8)

    # plt.title('PCA using modeltype: ' + self.hps['rnn_type'])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    # plt.show()
    return fig, ax


def visualize_flipflop(prediction, stim):
    """Function to visualize the 3-Bit flip flop task."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
    # plt.title('3-Bit Flip-Flop')
    ax1.xaxis.set_visible(False)
    ax2.xaxis.set_visible(False)

    ax1.plot(prediction[0, :, 0], c='r')
    ax1.plot(stim['inputs'][0, :, 0], c='k')
    ax2.plot(stim['inputs'][0, :, 1], c='k')
    ax2.plot(prediction[0, :, 1], c='g')
    ax3.plot(stim['inputs'][0, :, 2], c='k')
    ax3.plot(prediction[0, :, 2], c='b')
    plt.yticks([-1, +1])
    plt.xlabel('Time')

    plt.show()


class FixedPointPlotter(object):

    def __init__(self, activations, fps):

        self.activations = activations
        self.fps = fps

    def _extract_fixed_point_locations(self):
        """Processing of minimisation results for pca. The function takes one fixedpoint object at a time and
        puts all coordinates in single array."""
        fixed_point_location = [fp['x'] for fp in self.fps]

        fixed_point_locations = np.vstack(fixed_point_location)

        return fixed_point_locations

    def _classify_fixedpoints(self, scale):
        """Function to classify fixed points. Methodology is based on
        'Nonlinear Dynamics and Chaos, Strogatz 2015'.

        Args:
            fps: Fixedpointobject containing a set of fixedpoints.
            scale: Float by which the unstable modes shall be scaled for plotting.

        Returns:
            fps: Fixedpointobject that contains 'fp_stability', i.e. information about
            the stability of the fixedpoint
            x_directions: list of matrices containing vectors of unstable modes"""

        x_directions, fps = [], []
        scale = scale
        for fp in self.fps:

            trace = np.matrix.trace(fp['jac'])
            det = np.linalg.det(fp['jac'])
            if det > 0 and trace == 0:
                print('center has been found.')
            elif trace**2 - 4 * det == 0:
                print("star node has been found.")
            elif trace**2 - 4 * det < 0:
                print("spiral has been found")
            e_val, e_vecs = np.linalg.eig(fp['jac'])
            ids = np.argwhere(np.real(e_val) > 0)
            countgreaterzero = np.sum(e_val > 0)
            if countgreaterzero == 0:
                print('stable fixed point was found.')
                fp['fp_stability'] = 'stable fixed point'
                fps.append(fp)
            elif countgreaterzero > 0:
                print('saddle point was found.')
                fp['fp_stability'] = 'saddle point'
                fps.append(fp)
                for id in ids:
                    x_plus = fp['x'] + scale * e_val[id] * np.real(e_vecs[:, id].transpose())
                    x_minus = fp['x'] - scale * e_val[id] * np.real(e_vecs[:, id].transpose())
                    x_direction = np.vstack((x_plus, fp['x'], x_minus))
                    x_directions.append(np.real(x_direction))

        return fps, x_directions

    def plot_velocities(self, velocities, n_points: int = 5000):
        """This functions creates a 3D line plot of network activities
        colored by 'kinetic energy' i.e. rate of change in the dynamic system at
        any given position.

        Args:
            activations: Network activations. Must have dimensions [n_time, n_units].

            velocities: Vector of 'velocities' computed by feeding activations in q(x).
            Must have dimensions [n_time].

            n_points (optional): Number of points to be used for line plot. Recommended to choose values
             between 2000 and 5000. Default: 5000.

         Returns:
             None."""
        pca = skld.PCA(3)
        pca.fit(self.activations)
        X_pca = pca.transform(self.activations)

        mlab.plot3d(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
                    velocities[:n_points], line_width=0.2)
        mlab.colorbar(orientation='vertical')
        mlab.show()

    def plot_fixed_points(self, n_points, scale):

        fps, x_directions = self._classify_fixedpoints(scale)

        fixedpoints = self._extract_fixed_point_locations()
        if len(self.activations.shape) == 3:
            self.activations = np.vstack(self.activations)

        pca = skld.PCA(3)
        pca.fit(self.activations)
        X_pca = pca.transform(self.activations)
        new_pca = pca.transform(fixedpoints)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
                linewidth=0.7)
        for i in range(len(new_pca)):
            if fps[i]['fp_stability'] == 'stable fixed point':
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='k')
            elif fps[i]['fp_stability'] == 'saddle point':
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='r')
                for p in range(len(x_directions)):
                    direction_matrix = pca.transform(x_directions[p])
                    ax.plot(direction_matrix[:, 0], direction_matrix[:, 1], direction_matrix[:, 2],
                            c='r', linewidth=0.8)

        # plt.title('PCA using modeltype: ' + self.hps['rnn_type'])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        return fig, ax


class LanderFixedPointPlotter(FixedPointPlotter):

    def __init__(self, activations, fps, actions):

        FixedPointPlotter.__init__(self, activations, fps)

        self.actions = actions
        self.color_code = {'main_engine': 'r',
                           'left_engine': 'g',
                           'right_engine': 'b',
                           'main_left': 'c',
                           'main_right': 'y',
                           'no_engine': 'k'}
        self.definitive_actions = self._process_actions()

    def _process_actions(self):
        definitive_action = []

        for action in self.actions:
            if action[0] > 0 and np.abs(action[1]) > 0.5:
                if action[1] < -0.5:
                    definitive_action.append('main_left')
                elif action[1] > 0.5:
                    definitive_action.append('main_right')
            elif action[0] > 0:
                definitive_action.append('main_engine')
            elif action[1] < -0.5:
                definitive_action.append('left_engine')
            elif action[1] > 0.5:
                definitive_action.append('right_engine')
            else:
                definitive_action.append('no_engine')

        return definitive_action

    def plot_fixed_points(self, n_points, scale=1):

        fps, x_directions = self._classify_fixedpoints(scale)

        fixedpoints = self._extract_fixed_point_locations()
        if len(self.activations.shape) == 3:
            self.activations = np.vstack(self.activations)

        pca = skld.PCA(3)
        pca.fit(self.activations)
        X_pca = pca.transform(self.activations)
        new_pca = pca.transform(fixedpoints)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for k in range(n_points-1):
            ax.plot(X_pca[k:k+3, 0], X_pca[k:k+3, 1], X_pca[k:k+3, 2],
                    linewidth=0.7, c=self.color_code[self.definitive_actions[k]])

        for i in range(len(new_pca)):
            if fps[i]['fp_stability'] == 'stable fixed point':
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='k')
            elif fps[i]['fp_stability'] == 'saddle point':
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='r')
                for p in range(len(x_directions)):
                    direction_matrix = pca.transform(x_directions[p])
                    ax.plot(direction_matrix[:, 0], direction_matrix[:, 1], direction_matrix[:, 2],
                            c='r', linewidth=0.8)

        # plt.title('PCA using modeltype: ' + self.hps['rnn_type'])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.legend(self.color_code.keys(), loc='upper left')
        leg = ax.get_legend()
        for i, color in enumerate(self.color_code.items()):
            leg.legendHandles[i].set_color(color[1])
        return fig, ax


class MountainCarFixedPointPlotter(FixedPointPlotter):

    def __init__(self, activations, fps, actions):

        FixedPointPlotter.__init__(self, activations, fps)

        self.actions = actions
        self.color_code = {'left push': 'r',
                           'no push': 'k',
                           'right push': 'g'}
        self.definitive_actions = self._process_actions()

    def _process_actions(self):
        definitive_action = []

        for action in self.actions:
            if action == 0:
                definitive_action.append('left push')
            elif action == 1:
                definitive_action.append('no push')
            elif action == 2:
                definitive_action.append('right push')

        return definitive_action

    def plot_fixed_points(self, n_points, scale=1):

        fps, x_directions = self._classify_fixedpoints(scale)

        fixedpoints = self._extract_fixed_point_locations()
        if len(self.activations.shape) == 3:
            self.activations = np.vstack(self.activations)

        pca = skld.PCA(3)
        pca.fit(self.activations)
        X_pca = pca.transform(self.activations)
        new_pca = pca.transform(fixedpoints)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for k in range(n_points-3):
            ax.plot(X_pca[k:k+3, 0], X_pca[k:k+3, 1], X_pca[k:k+3, 2],
                    linewidth=0.7, c=self.color_code[self.definitive_actions[k]])

        for i in range(len(new_pca)):
            if fps[i]['fp_stability'] == 'stable fixed point':
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='k')
            elif fps[i]['fp_stability'] == 'saddle point':
                ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                           marker='.', s=30, c='r')
                for p in range(len(x_directions)):
                    direction_matrix = pca.transform(x_directions[p])
                    ax.plot(direction_matrix[:, 0], direction_matrix[:, 1], direction_matrix[:, 2],
                            c='r', linewidth=0.8)

        # plt.title('PCA using modeltype: ' + self.hps['rnn_type'])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.legend(self.color_code.keys(), loc='upper left')
        leg = ax.get_legend()
        for i, color in enumerate(self.color_code.items()):
            leg.legendHandles[i].set_color(color[1])
        return fig, ax