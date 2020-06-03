import autograd.numpy as np
import numdifftools as nd


class DynamicalSystemsBuilder(object):

    def __init__(self, recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        self.recurrent_layer_weights = recurrent_layer_weights
        self.n_recurrent_units = n_recurrent_units
        self.to_recurrent_layer_weights = to_recurrent_layer_weights
        self.from_recurrent_layer_weights = from_recurrent_layer_weights

    def build_ds(self, inputs):
        pass


class RnnDsBuilder(DynamicalSystemsBuilder):

    def __init__(self, recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        super().__init__(recurrent_layer_weights, n_recurrent_units,
                         to_recurrent_layer_weights, from_recurrent_layer_weights)

        self.weights, self.inputweights, self.b = recurrent_layer_weights[1], recurrent_layer_weights[0], \
                                                  recurrent_layer_weights[2]
        self.n_hidden = n_recurrent_units

    def build_sequential_ds(self, inputs):
        def fun(x):
            return 1 / self.n_hidden * np.sum((- x + np.tanh(x @ self.weights + inputs @ self.inputweights + self.b)) ** 2)

        return fun

    def build_joint_ds(self, inputs):
        def fun(x):
            return np.mean(
                1 / self.n_hidden * np.sum(((- x + np.tanh(x @ self.weights + inputs @ self.inputweights + self.b)) ** 2),
                                           axis=1))

        return fun

    def build_velocity_fun(self, inputs):
        def fun(x):
            return 0.5 * np.sum(((- x + np.tanh(x @ self.weights + inputs @ self.inputweights + self.b)) ** 2), axis=1)

        return fun

    def build_jacobian_fun(self, input):
        jac_fun = lambda x: - np.eye(self.n_hidden, self.n_hidden) + self.weights * (1 - np.tanh(x) ** 2)

        return jac_fun


class GruDsBuilder(DynamicalSystemsBuilder):

    def __init__(self, recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        super().__init__(recurrent_layer_weights, n_recurrent_units,
                         to_recurrent_layer_weights, from_recurrent_layer_weights)

        weights, self.n_hidden = self.recurrent_layer_weights, n_recurrent_units

        z, r, h = np.arange(0, self.n_hidden), np.arange(self.n_hidden, 2 * self.n_hidden), \
                  np.arange(2 * self.n_hidden, 3 * self.n_hidden)
        self.W_z, self.W_r, self.W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
        self.U_z, self.U_r, self.U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
        self.b_z, self.b_r, self.b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def _get_projections(self, input):
        self.z_projection_b = input @ self.W_z + self.b_z
        self.r_projection_b = input @ self.W_r + self.b_r
        self.g_projection_b = input @ self.W_h + self.b_h

    def build_joint_ds(self, input):
        self._get_projections(input)

        z_fun = lambda x: self.sigmoid(x @ self.U_z + self.z_projection_b)
        r_fun = lambda x: self.sigmoid(x @ self.U_r + self.r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + self.g_projection_b)

        def fun(x):
            return np.mean(1 / self.n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2), axis=1))

        return fun

    def build_sequential_ds(self, input):
        self._get_projections(input)

        z_fun = lambda x: self.sigmoid(x @ self.U_z + self.z_projection_b)
        r_fun = lambda x: self.sigmoid(x @ self.U_r + self.r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + self.g_projection_b)

        def fun(x):
            return 1/self.n_hidden * np.sum((- x + (z_fun(x) * x) + ((1 - z_fun(x)) * g_fun(x))) ** 2)

        return fun

    def build_velocity_fun(self, input):
        self._get_projections(input)

        z_fun = lambda x: self.sigmoid(x @ self.U_z + self.z_projection_b)
        r_fun = lambda x: self.sigmoid(x @ self.U_r + self.r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + self.g_projection_b)

        fun = lambda x: 1/self.n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2), axis=1)

        return fun

    def build_jacobian_fun(self, input):
        self._get_projections(input)

        z_fun = lambda x: self.sigmoid(x @ self.U_z + self.z_projection_b)
        r_fun = lambda x: self.sigmoid(x @ self.U_r + self.r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + self.g_projection_b)

        def dynamical_system(x):
            return - x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)
        jac_fun = nd.Jacobian(dynamical_system)

        return jac_fun


class LstmDsBuilder(DynamicalSystemsBuilder):

    def __init__(self, recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        super().__init__(recurrent_layer_weights, n_recurrent_units,
                         to_recurrent_layer_weights, from_recurrent_layer_weights)

        weights, self.n_hidden = self.recurrent_layer_weights, self.n_recurrent_units
        W, U, b = weights[0], weights[1], weights[2]

        self.W_i, self.W_f, self.W_c, self.W_o = W[:, :self.n_hidden], W[:, self.n_hidden:2 * self.n_hidden], \
                             W[:, 2 * self.n_hidden:3 * self.n_hidden], W[:, 3 * self.n_hidden:]
        self.U_i, self.U_f, self.U_c, self.U_o = U[:, :self.n_hidden], U[:, self.n_hidden:2 * self.n_hidden], \
                             U[:, 2 * self.n_hidden:3 * self.n_hidden], U[:, 3 * self.n_hidden:]
        self.b_i, self.b_f, self.b_c, self.b_o = b[:self.n_hidden], b[self.n_hidden:2 * self.n_hidden], \
                             b[2 * self.n_hidden:3 * self.n_hidden], b[3 * self.n_hidden:]

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def _get_projections(self, inputs):
        self.f_projection_b = inputs @ self.W_f + self.b_f
        self.i_projection_b = inputs @ self.W_i + self.b_i
        self.o_projection_b = inputs @ self.W_o + self.b_o
        self.c_projection_b = inputs @ self.W_c + self.b_c

    def build_joint_ds(self, input):
        self._get_projections(input)

        f_fun = lambda x: self.sigmoid(x @ self.U_f + self.f_projection_b)
        i_fun = lambda x: self.sigmoid(x @ self.U_i + self.i_projection_b)
        o_fun = lambda x: self.sigmoid(x @ self.U_o + self.o_projection_b)
        c_fun = lambda c, h: f_fun(h) * c + i_fun(h) * np.tanh((h @ self.U_c + self.c_projection_b))


        def h_fun(x):
            c, h = x[:, self.n_hidden:], x[:, :self.n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[:, self.n_hidden:], x[:, :self.n_hidden]
            return c_fun(c, h)

        def fun(x):
            return np.mean(1 / self.n_hidden * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x), axis=1))

        return fun

    def build_sequential_ds(self, inputs):
        self._get_projections(inputs)

        f_fun = lambda x: self.sigmoid(x @ self.U_f + self.f_projection_b)
        i_fun = lambda x: self.sigmoid(x @ self.U_i + self.i_projection_b)
        o_fun = lambda x: self.sigmoid(x @ self.U_o + self.o_projection_b)
        c_fun = lambda c, h: f_fun(h) * c + i_fun(h) * np.tanh((h @ self.U_c + self.c_projection_b))

        def h_fun(x):
            c, h = x[self.n_hidden:], x[:self.n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[self.n_hidden:], x[:self.n_hidden]
            return c_fun(c, h)

        def fun(x):
            return 0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x))

        return fun

    def build_velocity_fun(self, inputs):
        self._get_projections(inputs)

        f_fun = lambda x: self.sigmoid(x @ self.U_f + self.f_projection_b)
        i_fun = lambda x: self.sigmoid(x @ self.U_i + self.i_projection_b)
        o_fun = lambda x: self.sigmoid(x @ self.U_o + self.o_projection_b)
        c_fun = lambda c, h: f_fun(h) * c + i_fun(h) * np.tanh((h @ self.U_c + self.c_projection_b))

        def h_fun(x):
            c, h = x[:, self.n_hidden:], x[:, :self.n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[:, self.n_hidden:], x[:, :self.n_hidden]
            return c_fun(c, h)

        def fun(x):
            return 0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x), axis=1)

        return fun

    def build_jacobian_fun(self, inputs):
        self._get_projections(inputs)

        f_fun = lambda x: self.sigmoid(x @ self.U_f + self.f_projection_b)
        i_fun = lambda x: self.sigmoid(x @ self.U_i + self.i_projection_b)
        o_fun = lambda x: self.sigmoid(x @ self.U_o + self.o_projection_b)
        c_fun = lambda c, h: f_fun(h) * c + i_fun(h) * np.tanh((h @ self.U_c + self.c_projection_b))

        def h_fun(x):
            c, h = x[:, self.n_hidden:], x[:, :self.n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[:, self.n_hidden:], x[:, :self.n_hidden]
            return c_fun(c, h)

        def dynamical_system(x):
            return np.hstack((h_fun(x), cfun(x))) - x

        jac_fun = nd.Jacobian(dynamical_system)

        return jac_fun


class CircularGruBuilder(GruDsBuilder):

    def __init__(self, recurrent_layer_weights, n_recurrent_units, env, act_state_weights,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        super().__init__(recurrent_layer_weights, n_recurrent_units,
                         to_recurrent_layer_weights, from_recurrent_layer_weights)
        self.env = env
        self.act_state_weights = act_state_weights

    def build_numpy_submodelto(self):
        first_layer = lambda x: np.tanh(x @ self.to_recurrent_layer_weights[0] + self.to_recurrent_layer_weights[1])
        second_layer = lambda x: np.tanh(x @ self.to_recurrent_layer_weights[2] + self.to_recurrent_layer_weights[3])
        third_layer = lambda x: np.tanh(x @ self.to_recurrent_layer_weights[4] + self.to_recurrent_layer_weights[5])
        fourth_layer = lambda x: np.tanh(x @ self.to_recurrent_layer_weights[6] + self.to_recurrent_layer_weights[7])

        return first_layer, second_layer, third_layer, fourth_layer

    def build_numpy_submodelfrom(self):
        softplus = lambda x: np.log(np.exp(x) + 1)
        alpha_fun = lambda x: softplus(x @ self.from_recurrent_layer_weights[0] + self.from_recurrent_layer_weights[1])
        beta_fun = lambda x: softplus(x @ self.from_recurrent_layer_weights[2] + self.from_recurrent_layer_weights[3])

        return alpha_fun, beta_fun

    def act_deterministic(self, alphas, betas):
        actions = (alphas - 1) / (alphas + betas - 2)

        action_max_values = self.env.action_space.high
        action_min_values = self.env.action_space.low
        action_mm_diff = action_max_values - action_min_values

        actions = np.multiply(actions, action_mm_diff) + action_min_values

        return actions

    def build_circular_joint_model(self):
        first_layer, second_layer, third_layer, fourth_layer = self.build_numpy_submodelto()
        alpha_fun, beta_fun = self.build_numpy_submodelfrom()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        act_state_fun = lambda x: self.act_deterministic(alpha_fun(x), beta_fun(x)) @ self.act_state_weights
        f_fun = lambda x: fourth_layer(third_layer(second_layer(first_layer(act_state_fun(x)))))

        z_fun = lambda x: sigmoid(x @ self.U_z + f_fun(x) @ self.W_z + self.b_z)
        r_fun = lambda x: sigmoid(x @ self.U_r + f_fun(x) @ self.W_r + self.b_r)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + f_fun(x) @ self.W_h + self.b_h)

        def fun(x):
            return np.mean(1 / self.n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2), axis=1))

        return fun

    def build_circular_sequential_model(self):
        first_layer, second_layer, third_layer, fourth_layer = self.build_numpy_submodelto()
        alpha_fun, beta_fun = self.build_numpy_submodelfrom()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        act_state_fun = lambda x: self.act_deterministic(alpha_fun(x), beta_fun(x)) @ self.act_state_weights
        f_fun = lambda x: fourth_layer(third_layer(second_layer(first_layer(act_state_fun(x)))))

        z_fun = lambda x: sigmoid(x @ self.U_z + f_fun(x) @ self.W_z + self.b_z)
        r_fun = lambda x: sigmoid(x @ self.U_r + f_fun(x) @ self.W_r + self.b_r)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + f_fun(x) @ self.W_h + self.b_h)

        def fun(x):
            return 1 / self.n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2))

        return fun

    def build_jacobian_fun(self):
        first_layer, second_layer, third_layer, fourth_layer = self.build_numpy_submodelto()
        alpha_fun, beta_fun = self.build_numpy_submodelfrom()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        act_state_fun = lambda x: self.act_deterministic(alpha_fun(x), beta_fun(x)) @ self.act_state_weights
        f_fun = lambda x: fourth_layer(third_layer(second_layer(first_layer(act_state_fun(x)))))

        z_fun = lambda x: sigmoid(x @ self.U_z + f_fun(x) @ self.W_z + self.b_z)
        r_fun = lambda x: sigmoid(x @ self.U_r + f_fun(x) @ self.W_r + self.b_r)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + f_fun(x) @ self.W_h + self.b_h)

        def dynamical_system(x):
            return - x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)
        jac_fun = nd.Jacobian(dynamical_system)

        return jac_fun


class HopfDsBuilder:

    def __init__(self, weights, n_hidden):
        # parameters
        self.f = .05  # intrinsic frequency
        self.G = .1  # coupling strength
        self.a1 = .1  # bifurcation parameter of region 1
        self.a2 = .2  # bifurcation parameter of region 2
        self.beta = .02  # scaling factor of noise

        # coupling
        self.C = np.zeros((2, 2))  # adjacency matrix
        self.C[(1, 0)] = 1.  # region 1 affects region 2
        self.C[(0, 1)] = 1.  # region 2 affects region 1

        # setup
        self.w = self.f * 2. * np.pi  # angular frequency
        self.A = [self.a1, self.a2]
        self.dsig = 0. #np.sqrt(1)*self.beta # precalculated timestep for noise

    def build_joint_ds(self, inputs):
        DiffX = lambda X: np.tile(X, (2, 1)) - np.transpose(np.tile(X, (2, 1)))
        dx = lambda X, Y: - X + (
                    (self.A - pow(X, 2) - pow(Y, 2)) * X - self.w * Y + self.G * (np.sum(self.C * DiffX(X), axis=1))) \
                          + self.dsig * np.random.randn(2)
        DiffY = lambda Y: np.tile(Y, (2, 1)) - np.transpose(np.tile(Y, (2, 1)))
        dy = lambda X, Y: - Y + (
                    (self.A - pow(X, 2) - pow(Y, 2)) * Y + self.w * X + self.G * (
                np.sum(self.C * DiffY(Y), axis=1))) + self.dsig * np.random.randn(2)

        def fun(X):
            x, y = X[:2], X[2:]
            return np.sum(np.abs((dx(x,y), dy(x,y))))

        return fun

    def build_sequential_ds(self, inputs):
        DiffX = lambda X: np.tile(X, (2, 1)) - np.transpose(np.tile(X, (2, 1)))
        dx = lambda X, Y: - X + (
                    (self.A - pow(X, 2) - pow(Y, 2)) * X - self.w * Y + self.G * (np.sum(self.C * DiffX(X), axis=1))) \
                          + self.dsig * np.random.randn(2)
        DiffY = lambda Y: np.tile(Y, (2, 1)) - np.transpose(np.tile(Y, (2, 1)))
        dy = lambda X, Y: - Y + (
                    (self.A - pow(X, 2) - pow(Y, 2)) * Y + self.w * X + self.G * (
                np.sum(self.C * DiffY(Y), axis=1))) + self.dsig * np.random.randn(2)

        def fun(X):
            x, y = X[:2], X[2:]
            return np.sum(np.abs((dx(x,y), dy(x,y))))

        return fun

    def build_jacobian_fun(self, inputs):
        DiffX = lambda X: np.tile(X, (2, 1)) - np.transpose(np.tile(X, (2, 1)))
        dx = lambda X, Y: - X + (
                    (self.A - pow(X, 2) - pow(Y, 2)) * X - self.w * Y + self.G * (np.sum(self.C * DiffX(X), axis=1))) \
                          + self.dsig * np.random.randn(2)
        DiffY = lambda Y: np.tile(Y, (2, 1)) - np.transpose(np.tile(Y, (2, 1)))
        dy = lambda X, Y: - Y + (
                    (self.A - pow(X, 2) - pow(Y, 2)) * Y + self.w * X + self.G * (
                np.sum(self.C * DiffY(Y), axis=1))) + self.dsig * np.random.randn(2)

        def fun(X):
            x, y = X[:2], X[2:]
            return np.sum((dx(x,y), dy(x,y)))
        jac_fun = nd.Jacobian(fun)
        return jac_fun




