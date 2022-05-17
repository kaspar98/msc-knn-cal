import numpy as np


def generate_data(dirichlet, n_data, calibration_function, random_seed=None):

    np.random.seed(random_seed)

    p = np.random.dirichlet(dirichlet, n_data)

    c = calibration_function(p)
    c = c / np.sum(c, axis=1).reshape(-1, 1)

    y = np.array([np.random.choice(np.arange(0, len(dirichlet)), p=pred) for pred in c])
    y = np.eye(len(dirichlet))[y]

    return {"p": p, "c": c, "y": y}
