import numpy as np

def generate_normal_cost_matrices(n_classes, n_decisions, n_matrices, seed, mean=0, std=1):
    np.random.seed(seed)
    costs = np.random.normal(mean, std, n_classes*n_decisions*n_matrices)
    return costs.reshape((n_matrices, n_decisions, n_classes))

def generate_exp_cost_matrices(n_classes, n_decisions, n_matrices, seed, mean=0):
    np.random.seed(seed)
    costs = np.random.exponential(mean, n_classes*n_decisions*n_matrices)
    return costs.reshape((n_matrices, n_decisions, n_classes))

def true_loss_predicted_loss(p, y, cost_matrices):
    optimal_decisions = np.argmin(np.einsum("pc,mdc->mpd", p, cost_matrices), axis=2)
    pred_loss = np.mean(np.sum(cost_matrices[np.arange(cost_matrices.shape[0])[:, None], optimal_decisions] * p, axis=2), axis=1)
    true_loss = np.mean(np.sum(cost_matrices[np.arange(cost_matrices.shape[0])[:, None], optimal_decisions] * y, axis=2), axis=1)
    return true_loss, pred_loss

def loss_gap(p, y, n_classes, n_decisions, n_matrices, seed=0, mean=0, std=1):
    cost_matrices = generate_normal_cost_matrices(n_classes=n_classes, n_decisions=n_decisions,
                                                  n_matrices=n_matrices, seed=seed, mean=mean, std=std)

    optimal_decisions = np.argmin(np.einsum("pc,mdc->mpd", p, cost_matrices), axis=2)
    pred_loss = np.mean(np.sum(cost_matrices[np.arange(n_matrices)[:, None], optimal_decisions] * p, axis=2), axis=1)
    true_loss = np.mean(np.sum(cost_matrices[np.arange(n_matrices)[:, None], optimal_decisions] * y, axis=2), axis=1)
    loss_gap = np.abs(true_loss - pred_loss) #/ np.max(np.sum(cost_matrices**2, axis=2)**0.5, axis=1)
    return loss_gap

def optimal_decisions(predictions, cost_matrix):
    n_decisions = len(cost_matrix)
    n_classes = len(cost_matrix[0])
    n_preds = len(predictions)

    return ((np.tile(predictions, n_decisions) * cost_matrix.flatten())
            .reshape((n_preds, n_decisions, n_classes))
            .sum(axis=2)
            .argmin(axis=1)
            )

def mean_cost_of_decisions(cost_matrix, decisions, y):
    return np.mean(np.sum(cost_matrix[decisions] * y, axis=1))
