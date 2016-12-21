import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class SoftmaxClassifier(object):
    '''
    Basic implementation of a softmax classifier.

    Parameters
    ----------
    random_state: int (default: 1337)
        Defines the random state of numpy for a particular class instance.
    n_classes: int (default: 2)
        Number of classes used for the classification.
    epochs: int (default: 500)
        Number of epochs to train the model. This should be validated with a
        validation set in real world problems to avoid overtraining or
        stopping too early.
    learning_rate: float (default: 0.1)
        Initial learning rate. Also has to be validated to archieve a fast
        convergence. A learning rate decay schedule still needs to be
        implemented.
    '''
    def __init__(self,
                 random_state=None,
                 n_classes=2,
                 epochs=500,
                 learning_rate=.1):
        if random_state is None:
            np.random.RandomState(1337)
        else:
            np.random.RandomState(random_state)
        self._is_fitted = False
        self._W = None
        self._b = None
        self._n_classes = n_classes
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, Xs, Ys, init_params=True):
        '''
        Fits the model for given `Xs` and `Ys` and initializes all the
        parameters if needed.

        Parameters
        ----------
        Xs: array of shape (n_points, n_features)
            Two dimensional array containing data points with all features.
        Ys: array of shape (n_points, n_classes)
            Two dimensional array containing the labels of all data point in a
            one-hot structure.
        init_params: bool
            If True the weights and biases get initialized, otherwise the
            class attributes _W and _b are used.
        '''
        self._fit(Xs=Xs, Ys=Ys, init_params=init_params)
        self._is_fitted = True
        return self

    def _fit(self, Xs, Ys, init_params):
        if init_params:
            n_classes = self._n_classes
            n_features = Xs.shape[1]
            self._W, self._b = self._init_params(n_features, n_classes)
            self.costs = []

        if self._W.any() is None or self._b.any() is None:
            raise AttributeError(
                'Initialize weights and biases before fitting.')

        # Implement batch use here! For the moment one batch = all data
        for i in range(self.epochs):
            scores = self._scores(Xs, self._W, self._b)
            probs = self._softmax(scores)
            cost = self._cost(probs, Ys)
            self.costs.append(cost)
            dW, db = self._calc_gradient(Xs, Ys, probs)

            # The gradient can also be calculated numerically to check
            # the analytical calculation
            # dW_, db_ = self._calc_numeric_gradient(Xs, Ys)
            # print(dW, db)
            # print(dW_, db_)

            self._W += -self.learning_rate * dW
            self._b += -self.learning_rate * db

    def predict(self, Xs):
        '''
        Predict the class of data points `Xs` given the current model.

        Parameters
        ----------
        Xs: array of shape (n_points, n_features)
            Two dimensional array containing data points with all features.

        Returns
        -------
        array of shape (n_points,)
            The returned array contains the predicted classlabel for every data
            point in `Xs`.
        '''
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet!')
        return self._predict(Xs)

    def _predict(self, Xs):
        scores = self._scores(Xs, self._W, self._b)
        probs = self._softmax(scores)
        return self._to_classlabel(probs)

    def _calc_gradient(self, Xs, Ys, probs):
        diff = -(Ys - probs)
        diff /= diff.shape[0]
        dW = Xs.T.dot(diff)
        db = np.sum(diff, axis=0)
        return dW, db

    def _calc_numeric_gradient(self, Xs, Ys):
        from copy import deepcopy as cp
        dW = np.zeros(self._W.shape)
        db = np.zeros(self._b.shape)

        h = self.learning_rate

        it = np.nditer(self._W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index

            Wph = cp(self._W)
            Wph[ix] += h
            yp_pred = self._softmax(self._scores(Xs, Wph, self._b))
            cph = self._cost(yp_pred, Ys)

            Wmh = cp(self._W)
            Wmh[ix] -= h
            ym_pred = self._softmax(self._scores(Xs, Wmh, self._b))
            cmh = self._cost(ym_pred, Ys)

            dW[ix] = (cph - cmh) / (2 * h)
            it.iternext()

        it = np.nditer(self._b, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            bph = cp(self._b)
            bph[ix] += h
            yp_pred = self._softmax(self._scores(Xs, self._W, bph))
            cph = self._cost(yp_pred, Ys)

            bmh = cp(self._b)
            bmh[ix] -= h
            ym_pred = self._softmax(self._scores(Xs, self._W, bmh))
            cmh = self._cost(ym_pred, Ys)

            db[ix] = (cph - cmh) / (2 * h)
            it.iternext()

        return dW, db

    def _init_params(self, n_features, n_classes):
        weight_shape = (n_features, n_classes)
        bias_shape = (n_classes,)
        W = np.random.rand(n_features * n_classes) / \
            np.sqrt(n_features * n_classes)
        W = W.reshape(weight_shape)
        b = np.zeros(bias_shape)
        return W, b

    def _scores(self, xs, W, b):
        scores = np.matmul(xs, W) + b
        return scores

    def _softmax(self, scores):
        scores -= np.max(scores)
        probs = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
        return probs

    def _to_classlabel(self, probs):
        return np.argmax(probs, axis=1)

    def _cross_entropy(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-10, 1)
        cross_entropy = -np.sum(y_true * np.log(y_pred), axis=1)
        return cross_entropy

    def _cost(self, y_pred, y_true):
        '''
        This function calculates the actual cost,
        which allows to add additional terms to the function,
        e.g. L2 regularization.
        '''
        return np.mean(self._cross_entropy(y_pred, y_true))

    def generate_easy_data_(self, N_points=50):
        Xs_0_mean = [2., 10.]
        Xs_0_cov = [[1., 0.], [0., 1.]]
        Xs_0 = np.random.multivariate_normal(Xs_0_mean,
                                             Xs_0_cov,
                                             size=N_points)

        Xs_1_mean = [5., 5.]
        Xs_1_cov = [[1., 0.], [0., 1.]]
        Xs_1 = np.random.multivariate_normal(Xs_1_mean,
                                             Xs_1_cov,
                                             size=N_points)

        Xs = np.append(Xs_0, Xs_1, axis=0)

        n_classes = 2
        Ys_0 = np.zeros(N_points, dtype=np.int)
        Ys_1 = np.ones(N_points, dtype=np.int)
        Ys_ = np.append(Ys_0, Ys_1)
        Ys = np.eye(n_classes)[Ys_]
        return Xs, Ys

    def generate_spiral_data(self, N_points=50):
        n_classes = self._n_classes
        n_features = 2
        Xs = np.zeros((N_points * n_classes, n_features))
        Ys = np.zeros((N_points * n_classes), dtype=np.int)
        for i in range(n_classes):
            ix = range(N_points * i, N_points * (i + 1))
            r = np.linspace(0., 1., N_points)
            t = np.linspace(i * 4, (i + 1) * 4, N_points) + \
                np.random.randn(N_points) * 0.2
            Xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Ys[ix] = i
        Ys = np.eye(n_classes)[Ys]
        return Xs, Ys

    def generate_two_pop(self):
        P0 = np.load('P0.npy')
        P1 = np.load('P1.npy')

        Xs = np.append(P0.T, P1.T, axis=0)
        Ys = np.append(np.zeros(P0.shape[1], dtype=np.int),
                       np.ones(P1.shape[1], dtype=np.int))
        Ys = np.eye(2)[Ys]
        return Xs, Ys


def plot_decision_regions(Xs, Ys, classifier,
                          ax=None, res=.01, colors=None):
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = sns.color_palette()

    x_min, x_max = np.min(Xs[:, 0]) - .1, np.max(Xs[:, 0]) + .1
    y_min, y_max = np.min(Xs[:, 1]) - .1, np.max(Xs[:, 1]) + .1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z,
                alpha=.3,
                levels=np.arange(Z.max() + 2) - .5,
                colors=colors)
    ax.axis(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max)

    for ix in np.unique(np.argmax(Ys, axis=1)):
        ax.scatter(x=Xs[np.argmax(Ys, axis=1) == ix, 0],
                   y=Xs[np.argmax(Ys, axis=1) == ix, 1],
                   alpha=.8,
                   label='Class {}'.format(int(ix)),
                   c=colors[ix])
    ax.legend(loc='lower right', fancybox=True)


if __name__ == '__main__':
    sc = SoftmaxClassifier()
    Xs, Ys = sc.generate_two_pop()
    sc.fit(Xs, Ys)
    plot_decision_regions(Xs, Ys, sc)
    plt.savefig('2_pop.pdf')

    sc = SoftmaxClassifier(n_classes=3)
    Xs, Ys = sc.generate_spiral_data()
    sc.fit(Xs, Ys)
    plot_decision_regions(Xs, Ys, sc)
    plt.savefig('spiral_data.pdf')
