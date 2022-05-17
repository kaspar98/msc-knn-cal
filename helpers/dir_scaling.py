import tensorflow
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import keras.backend as K
from sklearn.model_selection import KFold


def calibrate_with_ms_pretuned(x_train, y_train, x_test, dataset):
    # Optimal parameters provided by authors of https://arxiv.org/abs/1910.12656
    l2_dict = {"densenet40_c10": 250,
               "densenet40_c100": 2.5,
               "lenet5_c10": 0.1,
               "lenet5_c100": 0.25,
               "resnet_wide32_c10": 0.5,
               "resnet_wide32_c100": 2.5,
               "resnet110_c10": 0.25,
               "resnet110_c100": 0.5,
               }
    mu_dict = {"densenet40_c10": 0.0001,
               "densenet40_c100": 10_000,
               "lenet5_c10": 0.00001,
               "lenet5_c100": 0.01,
               "resnet_wide32_c10": 0.00001,
               "resnet_wide32_c100": 0.01,
               "resnet110_c10": 0.00001,
               "resnet110_c100": 0.01,
               }

    ms = Dirichlet_NN(l2=l2_dict[dataset], mu=mu_dict[dataset], patience=50, use_logits=True)
    ms.fit(x_train, y_train.argmax(axis=1), verbose=True)
    cal_p_test = ms.predict(x_test)
    cal_p_train = ms.predict(x_train)
    return cal_p_test, cal_p_train, None

def calibrate_with_dir_pretuned(p_train, y_train, p_test, dataset):
    # Optimal parameters provided by authors of https://arxiv.org/abs/1910.12656
    l2_dict = {"densenet40_c10": 1000,
               "densenet40_c100": 5000,
               "lenet5_c10": 0.25,
               "lenet5_c100": 0.25,
               "resnet_wide32_c10": 0.5,
               "resnet_wide32_c100": 5000,
               "resnet110_c10": 0.25,
               "resnet110_c100": 5000,
               }
    mu_dict = {"densenet40_c10": 0.01,
               "densenet40_c100": 0.01,
               "lenet5_c10": 0.00001,
               "lenet5_c100": 1000000,
               "resnet_wide32_c10": 0.00001,
               "resnet_wide32_c100": 0.01,
               "resnet110_c10": 0.001,
               "resnet110_c100": 0.01,
               }

    dir = Dirichlet_NN(l2=l2_dict[dataset], mu=mu_dict[dataset], patience=50, use_logits=False)
    dir.fit(p_train, y_train.argmax(axis=1), verbose=True)
    cal_p_test = dir.predict(p_test)
    cal_p_train = dir.predict(p_train)
    return cal_p_test, cal_p_train, None


def calibrate_preds_with_dir_scaling(x_train, y_train, x_test, params_to_try, n_cv_folds, cv_loss, use_logits):
    cv_scores = find_dir_reg_param_with_cv(p=x_train, y=y_train, params_to_try=params_to_try, n_cv_folds=n_cv_folds,
                                           cv_loss=cv_loss, use_logits=use_logits)

    print(cv_scores)
    params_to_use = params_to_try[np.argmin(cv_scores)]

    dir = Dirichlet_NN(l2=params_to_use[0], mu=params_to_use[1], patience=50, use_logits=use_logits)
    dir.fit(x_train, y_train.argmax(axis=1), verbose=True)
    cal_p_test = dir.predict(x_test)
    cal_p_train = dir.predict(x_train)

    return cal_p_test, cal_p_train, cv_scores


def find_dir_reg_param_with_cv(p, y, params_to_try, n_cv_folds, cv_loss, use_logits):
    param_scores = np.zeros(len(params_to_try))

    kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(p):
        print("fold start")
        p_train, p_test = p[train_index], p[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for idx, (l2, mu) in enumerate(params_to_try):
            print("param test")
            dir = Dirichlet_NN(l2=l2, mu=mu, use_logits=use_logits)
            dir.fit(p_train, y_train.argmax(axis=1))
            cal_p_test = dir.predict(p_test)

            param_scores[idx] += cv_loss(y_test, cal_p_test)

    param_scores = param_scores / n_cv_folds
    print(param_scores)

    return list(param_scores)

#########
#
#  Following code is from https://github.com/dirichletcal/experiments_dnn
#    Meelis Kull, Miquel Perelló-Nieto, Markus Kängsepp, Telmo de Menezes e
#    Silva Filho, Hao Song, and Peter A. Flach. Beyond temperature scaling: Obtaining
#    well-calibrated multiclass probabilities with dirichlet calibration. In NeurIPS, 2019.
#
#########

class Dirichlet_NN():

    def __init__(self, l2=0., mu=None, classes=-1, max_epochs=500, comp=True,
                 patience=15, lr=0.001, weights=[], random_state=15, loss="sparse_categorical_crossentropy",
                 double_fit=True, use_logits=False):
        """
        Initialize class

        Params:
            l2 (float): regularization for off-diag regularization.
            mu (float): regularization for bias. (if None, then it is set equal to lambda of L2)
            classes (int): how many classes in given data set. (based on logits)
            max_epochs (int): maximum iterations done by optimizer.
            comp (bool): whether use complementary (off_diag) regularization or not.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            weights (array): initial weights of model ([k,k], [k]) - weights + bias
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            double_fit (bool): fit twice the model, in the beginning with lr (default=0.001), and the second time 10x lower lr (lr/10)
            use_logits (bool): Using logits as input of model, leave out conversion to logarithmic scale.

        """

        self.max_epochs = max_epochs
        self.patience = patience
        self.classes = classes
        self.l2 = l2
        self.lr = lr
        self.weights = weights
        self.random_state = random_state
        self.loss = loss
        self.double_fit = double_fit
        self.use_logits = use_logits

        if mu:
            self.mu = mu
        else:
            self.mu = l2

        if comp:
            self.regularizer = self.L2_offdiag(l2=self.l2)
        else:
            self.regularizer = keras.regularizers.l2(l=self.l2)

        tensorflow.random.set_seed(random_state)

        if classes >= 1:
            self.model = self.create_model(classes, weights)
        else:
            self.model = None

        np.random.seed(random_state)

    def create_model(self, classes, weights=[], verbose=False):

        """
        Create model and add loss to it

        Params:
            classes (int): number of classes, used for input layer shape and output shape
            weights (array): starting weights in shape of ([k,k], [k]), (weights, bias)
            verbose (bool): whether to print out anything or not

        Returns:
            model (object): Keras model
        """

        model = Sequential()

        if not self.use_logits:  # Leave out converting to logarithmic scale if logits are used as input. #CHANGED not self.use_logits
            model.add(Lambda(self._logFunc, input_shape=[classes]))

            model.add(Dense(classes, activation="softmax"
                            , kernel_initializer=keras.initializers.Identity(gain=1)
                            , bias_initializer="zeros",
                            kernel_regularizer=self.regularizer, bias_regularizer=keras.regularizers.l2(l=self.mu)))

        else:
            model.add(Dense(classes, input_shape=[classes], activation="softmax"
                            , kernel_initializer=keras.initializers.Identity(gain=1)
                            , bias_initializer="zeros",
                            kernel_regularizer=self.regularizer, bias_regularizer=keras.regularizers.l2(l=self.mu)))

        if len(weights) != 0:  # Weights that are set from fitting
            model.set_weights(weights)
        elif len(self.weights) != 0:  # Weights that are given from initialisation
            model.set_weights(self.weights)

        adam = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=adam)

        if verbose:
            model.summary()

        return model

    def fit(self, probs, true, weights=[], verbose=False, double_fit=None, batch_size=128):
        """
        Trains the model and finds optimal parameters

        Params:
            probs: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            weights (array): starting weights in shape of ([k,k], [k]), (weights, bias)
            verbose (bool): whether to print out anything or not
            double_fit (bool): fit twice the model, in the beginning with lr (default=0.001), and the second time 10x lower lr (lr/10)

        Returns:
            hist: Keras history of learning process
        """

        if len(weights) != 0:
            self.weights = weights

        if "sparse" not in self.loss:  # Check if need to make Y categorical; TODO Make it more see-through
            true = to_categorical(true)

        if double_fit == None:
            double_fit = self.double_fit

        self.model = self.create_model(probs.shape[1], self.weights, verbose)

        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]

        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size,
                              verbose=verbose)

        if double_fit:  # In case of my experiments it gave better results to start with default learning rate (0.001) and then fit again (0.0001) learning rate.
            if verbose:
                print("Fit with 10x smaller learning rate")
            self.lr = self.lr / 10
            self.fit(probs, true, weights=self.model.get_weights(), verbose=verbose, double_fit=False,
                     batch_size=batch_size)  # Fit 2 times

        return hist

    def predict(self, probs):  # TODO change it to return only the best prediction
        """
        Scales logits based on the model and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        return self.model.predict(probs)

    def predict_proba(self, probs):
        """
        Scales logits based on the model and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        return self.model.predict(probs)

    @property
    def coef_(self):
        """
        Actually weights of neurons, but to keep similar notation to original Dirichlet we name it coef_
        """
        if self.model:
            return self.model.get_weights()[0].T  # Transposed to match with full dirichlet weights.

    @property
    def intercept_(self):
        """
        Actually bias values, but to keep similar notation to original Dirichlet we name it intercept_
        """
        if self.model:
            return self.model.get_weights()[1]

    def _logFunc(self, x):
        """
        Find logarith of x (tensor)
        """
        eps = np.finfo(float).eps  # 1e-16

        return K.log(K.clip(x, eps, 1 - eps))  # How this clip works? K.clip(x, K.epsilon(), None) + 1.)

    # Inner classes for off diagonal regularization
    class Regularizer(object):
        """
        Regularizer base class.
        """

        def __call__(self, x):
            return 0.0

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    class L2_offdiag(Regularizer):
        """
        Regularizer for L2 regularization off diagonal.
        """

        def __init__(self, l2=0.0):
            """
            Params:
                l: (float) lambda, L2 regularization factor.
            """
            self.l2 = K.cast_to_floatx(l2)

        def __call__(self, x):
            """
            Off-diagonal regularization (complementary regularization)
            """

            reg = 0

            for i in range(0, x.shape[0]):
                reg += K.sum(self.l2 * K.square(x[0:i, i]))
                reg += K.sum(self.l2 * K.square(x[i + 1:, i]))

            return reg

        def get_config(self):
            return {'l2': float(self.l2)}
