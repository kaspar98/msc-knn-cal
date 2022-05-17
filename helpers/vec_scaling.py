import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def calibrate_preds_with_vs(logits_train, y_train, logits_test):
    vs = VectorScaling()
    vs.fit(logits_train, y_train)
    cal_p_test = vs.predict(logits_test)
    cal_p_train = vs.predict(logits_train)
    return cal_p_test, cal_p_train, None

#########
#
#  Following code is from https://github.com/dirichletcal/experiments_dnn
#    Meelis Kull, Miquel Perelló-Nieto, Markus Kängsepp, Telmo de Menezes e
#    Silva Filho, Hao Song, and Peter A. Flach. Beyond temperature scaling: Obtaining
#    well-calibrated multiclass probabilities with dirichlet calibration. In NeurIPS, 2019.
#
#########

class VectorScaling():

    def __init__(self, classes = 1, W = [], bias = [], maxiter = 100, solver = "BFGS", use_bias = True):
        """
        Initialize class

        Params:
            maxiter (int): maximum iterations done by optimizer.
            classes (int): how many classes in given data set. (based on logits )
            W (np.ndarray): matrix with temperatures for all the classes
            bias ( np.array): vector with biases
        """

        self.W = W
        self.bias = bias
        self.maxiter = maxiter
        self.solver = solver
        self.classes = classes
        self.use_bias = use_bias

    def _loss_fun(self, x, logits, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        W = np.diag(x[:self.classes])

        if self.use_bias:
            bias = x[self.classes:]
        else:
            bias = np.zeros(self.classes)
        scaled_probs = self.predict(logits, W, bias)

        loss = log_loss(true, scaled_probs)

        return loss

    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        self.classes = logits.shape[1]
        x0 = np.concatenate([np.repeat(1, self.classes), np.repeat(0, self.classes)])
        opt = minimize(self._loss_fun, x0 = x0, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.W = np.diag(opt.x[:logits.shape[1]])
        self.bias = opt.x[logits.shape[1]:]

        return opt

    def predict(self, logits, W = [], bias = []):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if len(W) == 0 or len(bias) == 0:  # Use class variables
            scaled_logits = np.dot(logits, self.W) + self.bias
        else:  # Take variables W and bias from arguments
            scaled_logits = np.dot(logits, W) + bias

        return softmax(scaled_logits)