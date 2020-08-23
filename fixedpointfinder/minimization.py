import autograd.numpy as np
from autograd import grad


class Minimizer(object):
    default_hps = {'epsilon': 1e-03,
                   'alr_decayr': 1e-03,
                   'max_iter': 5000,
                   'print_every': 200,
                   'init_agnc': 1,
                   'agnc_decayr': 1e-03,
                   'verbose': True}

    def __init__(self, epsilon=default_hps['epsilon'], alr_decayr=default_hps['alr_decayr'],
                       max_iter=default_hps['max_iter'], print_every=default_hps['print_every'],
                       init_agnc=default_hps['init_agnc'], agnc_decayr=default_hps['agnc_decayr'],
                       verbose=default_hps['verbose']):

        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-07
        self.epsilon = epsilon
        self.alr_decayr = alr_decayr
        self.max_iter = max_iter
        self.print_every = print_every
        self.init_agnc = init_agnc
        self.agnc_decayr = agnc_decayr
        self.verbose = verbose

    """Minimizer is a helper class containing functionality to minimize any abritrary vector valued or 
    scalar function. However, it is predominantly used by fixedpointfinder classes."""

    @staticmethod
    def _print_update(q, lr, agnc, norm):
        print(f"Function value: {np.round(q, 14)}| lr: {np.round(lr, 4)}| "
              f"agnc: {np.round(agnc,4)}| norm: {np.round(norm, 6)}")

    @staticmethod
    def _decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    @staticmethod
    def _decay_agnc(initial_clip, decay, iteration):
        return initial_clip * (1.0 / (1.0 + decay * iteration))

    def adam_optimization(self, fun, x0):
        """Function to implement the adam optimization algorithm. Also included in this function are
        functionality for adaptive learning rate as well as adaptive gradient norm clipping.

        Goal of this function is to find optimized activity of rnns."""

        beta_1, beta_2 = self.beta_1, self.beta_2
        eps = self.eps
        m, v = np.zeros(x0.shape), np.zeros(x0.shape)
        for t in range(self.max_iter):
            q = fun(x0)
            lr = self._decay_lr(self.epsilon, self.alr_decayr, t)
            agnc = self._decay_agnc(self.init_agnc, self.agnc_decayr, t)

            dq = grad(fun)(x0)
            norm = np.linalg.norm(dq)
            if norm > agnc:
                dq = dq / norm
            m = beta_1 * m + (1 - beta_1) * dq
            v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
            m_hat = m / (1 - np.power(beta_1, t + 1))
            v_hat = v / (1 - np.power(beta_2, t + 1))
            x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % self.print_every == 0 and self.verbose:
                self._print_update(q, lr, agnc, norm)

        return x0


def adam_lstm(fun, x0,
              epsilon, alr_decayr,
              max_iter, print_every,
              init_agnc, agnc_decayr,
              verbose):
    """Function to implement the adam optimization algorithm. Also included in this function are
    functionality for adaptive learning rate as well as adaptive gradient norm clipping."""
    def print_update(q, lr, norm):
        print("Function value:", q, "; lr", np.round(lr, 4), norm)

    def decay_lr(initial_lr, decay, iteration):
        return initial_lr * (1.0 / (1.0 + decay * iteration))

    def decay_agnc(initial_clip, decay, iteration):
        return initial_clip * (1.0 / (1.0 + decay * iteration))

    beta_1, beta_2 = 0.9, 0.999
    eps = 1e-08
    m, v = np.zeros(x0.shape), np.zeros(x0.shape)
    for t in range(max_iter):
        q = fun(x0)
        lr = decay_lr(epsilon, alr_decayr, t)
        agnc = decay_agnc(init_agnc, agnc_decayr, t)

        dq = grad(fun)(x0)

        norm_h = np.linalg.norm(dq)
        if norm_h > agnc:
            dq = dq / norm_h

        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t + 1))
        v_hat = v / (1 - np.power(beta_2, t + 1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + eps)

        if t % print_every == 0 and verbose:
            print_update(q, lr, norm_h)

    return x0


class RecordableMinimizer(Minimizer):

    def __init__(self, epsilon, alr_decayr,
                       max_iter, print_every,
                       init_agnc, agnc_decayr,
                       verbose):
        Minimizer.__init__(self, epsilon, alr_decayr,
                           max_iter, print_every,
                           init_agnc, agnc_decayr,
                           verbose)

    def adam_optimization(self, fun, x0):
        """Function to implement the adam optimization algorithm. Also included in this function are
        functionality for adaptive learning rate as well as adaptive gradient norm clipping."""

        optimization_history = []
        beta_1, beta_2 = self.beta_1, self.beta_2
        eps = self.eps
        m, v = np.zeros(x0.shape), np.zeros(x0.shape)
        for t in range(self.max_iter):
            q = fun(x0)
            lr = self._decay_lr(self.epsilon, self.alr_decayr, t)
            agnc = self._decay_agnc(self.init_agnc, self.agnc_decayr, t)

            dq = grad(fun)(x0)
            norm = np.linalg.norm(dq)
            if norm > agnc:
                dq = dq / norm
            m = beta_1 * m + (1 - beta_1) * dq
            v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
            m_hat = m / (1 - np.power(beta_1, t + 1))
            v_hat = v / (1 - np.power(beta_2, t + 1))
            x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % self.print_every == 0 and self.verbose:
                self._print_update(q, lr, agnc, norm)
                optimization_history.append(x0)

        return x0, optimization_history


class CircularMinimizer(Minimizer):

    def __init__(self,
                 epsilon, alr_decayr,
                 max_iter, print_every,
                 init_agnc, agnc_decayr,
                 verbose):

        Minimizer.__init__(self, epsilon, alr_decayr,
                           max_iter, print_every,
                           init_agnc, agnc_decayr,
                           verbose)

    def adam_optimization(self, fun, x0, t):
        """Function to implement the adam optimization algorithm. Also included in this function are
        functionality for adaptive learning rate as well as adaptive gradient norm clipping.

        Goal of this function is to find optimized activity of rnns."""

        beta_1, beta_2 = self.beta_1, self.beta_2
        eps = self.eps
        m, v = np.zeros(x0.shape), np.zeros(x0.shape)
        q = fun(x0)
        lr = self._decay_lr(self.epsilon, self.alr_decayr, t)
        agnc = self._decay_agnc(self.init_agnc, self.agnc_decayr, t)

        dq = grad(fun)(x0)
        norm = np.linalg.norm(dq)
        if norm > agnc:
            dq = dq / norm
        m = beta_1 * m + (1 - beta_1) * dq
        v = beta_2 * v + (1 - beta_2) * np.power(dq, 2)
        m_hat = m / (1 - np.power(beta_1, t + 1))
        v_hat = v / (1 - np.power(beta_2, t + 1))
        x0 = x0 - lr * m_hat / (np.sqrt(v_hat) + eps)

        if self.verbose:
            self._print_update(q, lr, agnc, norm)

        return x0


