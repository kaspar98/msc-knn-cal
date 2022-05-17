from scipy.optimize import minimize
from sklearn.metrics import log_loss
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def calibrate_preds_with_temps(logits_train, y_train, logits_test):
    ts = TemperatureScaling()
    ts.fit(logits_train, y_train.argmax(axis=1))
    cal_p_test = ts.predict(logits_test)
    cal_p_train = ts.predict(logits_train)
    return cal_p_test, cal_p_train, None

#########
#
#  Following code is from https://github.com/dirichletcal/experiments_dnn
#    Meelis Kull, Miquel Perelló-Nieto, Markus Kängsepp, Telmo de Menezes e
#    Silva Filho, Hao Song, and Peter A. Flach. Beyond temperature scaling: Obtaining
#    well-calibrated multiclass probabilities with dirichlet calibration. In NeurIPS, 2019.
#
#########

class TemperatureScaling():

    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
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

        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]

        return opt

    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)