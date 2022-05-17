#########
#
# Following code (with slight changes) is from the supplementary materials of https://proceedings.neurips.cc/paper/2021/hash/bbc92a647199b832ec90d7cf57074e9e-Abstract.html
#    Shengjia Zhao, Michael P Kim, Roshni Sahoo, Tengyu Ma, and Stefano Ermon.
#    Calibrating predictions to decisions: A novel approach to multi-class calibration.
#    In Thirty-Fifth Conference on Neural Information Processing Systems, 2021.
#
#########

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
import time
from .weighting_calibrators import force_p_to_simplex
from .proper_losses import bs

def calibrate_with_deccal(p_train, y_train, p_test, y_test, crop=1e-4, epochs=1000, calib_steps=100, num_action=2):

    calibrator = CalibratorDecision(verbose=True)
    calibrator.train(torch.from_numpy(p_train).to(torch.float),
                     torch.from_numpy(y_train.argmax(axis=1)).to(torch.int64),
                     calib_steps=calib_steps, num_action=num_action, norm=2,
                     num_critic_epoch=epochs,
                     test_predictions=torch.from_numpy(p_test).to(torch.float),
                     test_labels=torch.from_numpy(y_test.argmax(axis=1)).to(torch.int64))

    bs_te = []
    for i in range(1, 100):
        p = np.array(calibrator(torch.from_numpy(p_test).to(torch.float), max_critic=i))
        p = force_p_to_simplex(p, crop=crop)
        bs_te.append(bs(p, y_test))

    best_iter = np.argmin(bs_te)+1
    print(f"best iter {best_iter}")
    cal_p_test = np.array(calibrator(torch.from_numpy(p_test).to(torch.float), max_critic=best_iter))
    cal_p_test = force_p_to_simplex(cal_p_test, crop=crop)

    cal_p_train = np.array(calibrator(torch.from_numpy(p_train).to(torch.float), max_critic=best_iter))
    cal_p_train = force_p_to_simplex(cal_p_train, crop=crop)

    return cal_p_test, cal_p_train, None


def compute_decision_utility(predictions, labels, num_loss=200, num_action=2):
    device = predictions.device
    num_classes = predictions.shape[1]
    losses = torch.randn(num_loss, num_action, num_classes, device=device)
    losses.to(device)
    with torch.no_grad():
        pred_loss_all = torch.zeros(len(losses), device=device)  # Container for predicted loss
        true_loss_all = torch.zeros(len(losses), device=device)  # Container for true loss
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(predictions.cpu(), labels.cpu()),
                                                  batch_size=256, shuffle=False, num_workers=2)
        count = 0
        for i, data in enumerate(data_loader):
            bpred, blabels = data[0].to(device), data[1].to(device)
            # The predicted loss for each loss / sample / action
            pred_loss = (losses.view(len(losses), 1, num_action, num_classes) * bpred.reshape(1, -1, 1,
                                                                                              num_classes)).sum(
                dim=-1)  # number of losses, batch_size, number of actions
            bayes_loss, bayes_action = pred_loss.min(dim=-1)  # number of losses, batch_size
            pred_loss_all += bayes_loss.sum(dim=1)
            for k in range(bpred.shape[
                               0]):  # For each sample in the batch, compute the true loss of the bayes action (for each of the losses)
                true_loss_all += losses[range(len(losses)), bayes_action[:, k], blabels[k]]
    return pred_loss_all / len(predictions), true_loss_all / len(predictions)


def compute_accuracy(predictions, labels):
    return (torch.argmax(predictions, dim=1) == labels).type(torch.float32).mean()


class CriticDecision(nn.Module):
    def __init__(self, num_classes=1000, num_action=2):
        super(CriticDecision, self).__init__()
        self.fc = nn.Linear(num_classes, num_action, bias=False)
        self.adjustment = nn.Parameter(torch.zeros(num_action, num_classes), requires_grad=False)  # The adjustment for examples that belong to an action and for each class
        self.num_action = num_action
        self.num_classes = num_classes

    # Input the predicted probability (array of size [batch_size, number_of_classes]), and the labels (int array of size [batch_size])
    # Learn the optimal critic function and the new recalibration adjustment
    # norm should be 1 or 2
    def optimize(self, predictions, labels, num_epoch=20, norm=1):

        critic_optim = optim.Adam(self.fc.parameters(), lr=1e-2)
        lr_schedule = optim.lr_scheduler.StepLR(critic_optim, step_size=50, gamma=0.8)

        for epoch in range(num_epoch):

            # Direct computation
            for rep in range(10):
                critic_optim.zero_grad()
                loss = -self.evaluate_soft_diff(labels.detach(), predictions.detach(),
                                                noise=0.1).mean(dim=0, keepdim=True).norm(p=norm, dim=-1).sum()
                loss.backward()
                critic_optim.step()

            with torch.no_grad():
                adjustment = self.evaluate_adjustment(labels, predictions)
                self.adjustment = nn.Parameter(adjustment, requires_grad=False)
            lr_schedule.step()
        return loss

    # For each input probability output the (relaxed) probability of taking each action
    def forward(self, x, noise=0.0):
        fc = self.fc(x)
        fc = F.softmax(fc + torch.randn_like(fc) * noise, dim=1)
        return fc

    def evaluate_soft_diff(self, labels, pred_prob, noise=0.0):
        labels = F.one_hot(labels, num_classes=pred_prob.shape[1])  #[batch_size, num_classes]

        # print(z.shape, labels.shape)
        weights = self.forward(pred_prob, noise).view(-1, self.num_action, 1)   # shape should be batch_size, number of actions, 1
        diff = weights * (labels - pred_prob).view(-1, 1, pred_prob.shape[1])   # diff_{iaj} = (y_ij - \hat{p}(x_i)_j) softmax(<\hat{p}(x_i), l_a>
        return diff

    def evaluate_adjustment(self, labels, pred_prob):
        labels = F.one_hot(labels, num_classes=pred_prob.shape[1])  #[batch_size, num_classes]

        # print(z.shape, labels.shape)
        weights = self.forward(pred_prob, 0.0).unsqueeze(-1)   # shape should be batch_size, number of actions, 1
        diff = weights * (labels - pred_prob).view(-1, 1, pred_prob.shape[1])   # diff_{iaj} = (y_ij - \hat{p}(x_i)_j) softmax(<\hat{p}(x_i), l_a>
        coeff = torch.linalg.inv(torch.matmul(weights[:, :, 0].transpose(1, 0), weights[:, :, 0]))
        return torch.matmul(coeff, diff.sum(dim=0)).unsqueeze(0)

class CalibratorDecision():
    def __init__(self, verbose=False, device=None, save_path=None):
        self.critics = []
        self.verbose = verbose
        self.device = device
        self.save_path = save_path

    def __call__(self, x, max_critic=-1):
        for index, critic in enumerate(self.critics):
            if index == max_critic:
                break
            with torch.no_grad():
                # adjustment has shape [1, num_action, num_class], critic(x) has shape [batch_size, num_action]
                # print(critic.adjustment.shape)
                bias = (critic.adjustment * critic(x).unsqueeze(-1)).sum(dim=1) # bias should be [batch_size, num_class]
                # print(x.shape, bias.shape)
            x = x + bias   # Don't use inplace here
            # print(critic.evaluate_soft_diff(labels.detach(), predictions.detach()).mean(dim=0, keepdim=True).norm(p=2, dim=-1).sum())

        return x.clamp(min=1e-7, max=1-1e-7)

    def train(self, predictions, labels, calib_steps=200, num_action=2, num_critic_epoch=50, norm=1, test_predictions=None, test_labels=None):
        start_time = time.time()
        for step in range(calib_steps):
            with torch.no_grad():
                modified_prediction = self(predictions)
                pred_loss, true_loss = compute_decision_utility(modified_prediction.to(self.device),
                                                                labels.to(self.device),
                                                                num_action=num_action)
                accuracy = compute_accuracy(modified_prediction.to(self.device), labels.to(self.device))
                gap = pred_loss - true_loss
                print(f"Loss gap train: {gap.abs().mean().item()}")
                if test_predictions is not None:
                    modified_prediction = self(test_predictions)
                    pred_loss, true_loss = compute_decision_utility(modified_prediction.to(self.device),
                                                                    test_labels.to(self.device),
                                                                    num_action=num_action)
                    accuracy = compute_accuracy(modified_prediction.to(self.device), test_labels.to(self.device))
                    gap = pred_loss - true_loss
                    print(f"Loss gap test: {gap.abs().mean().item()}")

                if self.verbose:
                    print("Step %d, time=%.1f" % (step, time.time() - start_time))

            with torch.no_grad():
                updated_predictions = self(predictions)
                if test_predictions is not None:
                    updated_test_predictions = self(test_predictions)
                else:
                    updated_test_predictions = None
            critic = CriticDecision(num_action=num_action, num_classes=predictions.shape[1]).to(predictions.device)
            critic.optimize(predictions=updated_predictions, labels=labels, num_epoch=num_critic_epoch,
                             norm=norm)
            self.critics.append(critic)


            if step % 10 == 0 and self.save_path is not None:
                self.save(os.path.join(self.save_path,
                                       '%d-%d-%d.tar' % (predictions.shape[1], num_action, norm)))

    def save(self, save_path):
        if len(self.critics) == 0:
            return
        save_dict = {}
        for index, critic in enumerate(self.critics):
            save_dict[str(index)] = critic.state_dict()
        save_dict['num_action'] = self.critics[0].num_action
        save_dict['num_classes'] = self.critics[0].num_classes
        torch.save(save_dict, save_path)

    def load(self, save_path):
        self.critics = []
        loader = torch.load(save_path)
        print(len(loader))

        for index in range(len(loader) - 2):
            critic = CriticDecision(num_action=loader['num_action'], num_classes=loader['num_classes']).to(self.device)
            critic.load_state_dict(loader[str(index)])
            self.critics.append(critic)
