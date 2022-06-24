import os
import math
from re import X
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler
import torch
plt.style.use("seaborn-white")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_yaleb_test_contset(x, s, y):
    merged = np.concatenate((x, s.reshape(-1,1), y.reshape(-1,1)), axis=1)
    for y in range(38):
        for s in range(5):
            group_idx = np.logical_and((merged[:,-1] == y), (merged[:,-2]==s))
            globals()[f'ori_g{s}{y}'] = merged[group_idx]
            if (s == 0) and (y == 0):
                origin = globals()[f'ori_g{s}{y}']
            else:
                origin = np.concatenate((origin, globals()[f'ori_g{s}{y}']), axis=0)

    for y in range(38):
        for s in range(5):
            i = (s + random.randint(1, 4)) % 5
            # same id, different light group
            globals()[f'cont_g{s}{y}'] = globals()[f'ori_g{i}{y}']
            # ori, cont size matching
            globals()[f'cont_g{s}{y}'] = globals()[f'cont_g{s}{y}'][np.random.choice(range(len(globals()[f'cont_g{s}{y}'])), len(globals()[f'ori_g{s}{y}']), replace=True)]
            if (s == 0) and (y == 0):
                contrastive = globals()[f'cont_g{s}{y}']
            else:
                contrastive = np.concatenate((contrastive, globals()[f'cont_g{s}{y}']), axis=0)
    # release memory
    for y in range(38):
        for s in range(5):
            del globals()[f'cont_g{s}{y}']
            del globals()[f'ori_g{s}{y}']

    return origin, contrastive


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # no progress -> increase patience
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # renewal
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss change ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss