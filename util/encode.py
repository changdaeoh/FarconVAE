from dataset import *
import torch
import pandas as pd
import numpy as np


def encode_all(args, dataset, model, xy_clf, device, is_train=True):
    X = dataset.X
    s = dataset.s
    y = dataset.y
    s = pd.Series(s.reshape(-1), name=args.sensitive)
    y = pd.DataFrame(y, columns=[args.target])

    X_tensor, s_tensor, y_tensor = torch.FloatTensor(X), torch.FloatTensor(np.array(s)), torch.FloatTensor(np.array(y))
    X_tensor, s_tensor, y_tensor = X_tensor.to(device), s_tensor.to(device), y_tensor.to(device)
    model.eval()
    with torch.no_grad():
        if is_train:
            (zx, zs), _ = model.encode(X_tensor, s_tensor.reshape(-1, 1), y_tensor.reshape(-1, 1))
        else:
            raw_pred = xy_clf(X_tensor)
            y_hat = torch.ones_like(raw_pred, device=device)
            y_hat[raw_pred < 0] = 0.0    
            (zx, zs), _ = model.encode(X_tensor, s_tensor.reshape(-1, 1), y_hat)

    # array array series df
    return np.array(zx.cpu()), np.array(zs.cpu()), s, y


def encode_all_yaleb(args, model, xy_clf, device, is_train=True, data=0):
    if data == 0:
        data = ExtendedYaleBDataLoader(args.data_path, args=args, device=device)
        data.load()

    if is_train:
        s = data.train_sensitive_label
        s = F.one_hot(s, num_classes=5)
        X = data.train_data
        y = data.train_label
    else:
        s = data.test_sensitive_label
        s = F.one_hot(s, num_classes=5)
        X = data.test_data
        y = data.test_label

    X_tensor, s_tensor, y_tensor = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(s)), torch.FloatTensor(np.array(y))
    X_tensor, s_tensor, y_tensor = X_tensor.to(device), s_tensor.to(device), y_tensor.to(device)
    model.eval()
    with torch.no_grad():
        if is_train:
            (zx, zs), _ = model.encode(X_tensor, s_tensor, y_tensor, input_label=True)
        else:
            y_hat_clf_test = torch.argmax(xy_clf(X_tensor), dim=1)
            (zx, zs), _ = model.encode(X_tensor, s_tensor, y_hat_clf_test, input_label=True)
    s = pd.Series(np.argmax(s, axis=1), name='sensitive')

    # array, array, series, tensor
    return np.array(zx.cpu()), np.array(zs.cpu()), s, y


def odfr_encode(args, dataset, model, device):
    X = dataset.X
    s = dataset.s
    y = dataset.y
    s = pd.Series(s.reshape(-1), name=args.sensitive)
    y = pd.DataFrame(y, columns=[args.target])

    X_tensor = torch.FloatTensor(X)
    X_tensor = X_tensor.to(device)
    model.eval()
    with torch.no_grad():
        (zt, zs), _, _ = model(X_tensor)

    # array, array, series, DF
    return np.array(zt.cpu()), np.array(zs.cpu()), s, y


def odfr_encode_yaleb(args, model, device, is_train=True, data=0):
    if data == 0:
        data = ExtendedYaleBDataLoader(args.data_path, args=args, device=device)
        data.load()

    if is_train:
        s = data.train_sensitive_label
        s = F.one_hot(s, num_classes=5)
        X = data.train_data
        y = data.train_label
    else:
        s = data.test_sensitive_label
        s = F.one_hot(s, num_classes=5)
        X = data.test_data
        y = data.test_label

    X_tensor, s_tensor, y_tensor = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(s)), torch.FloatTensor(np.array(y))
    X_tensor, s_tensor, y_tensor = X_tensor.to(device), s_tensor.to(device), y_tensor.to(device)
    model.eval()
    with torch.no_grad():
        (zx, zs), _, _ = model(X_tensor)
    s = pd.Series(np.argmax(s, axis=1), name='sensitive')

    # array, array, series, tensor
    return np.array(zx.cpu()), np.array(zs.cpu()), s, y

def maxent_encode(args, dataset, model, device):
    X = dataset.X
    s = dataset.s
    y = dataset.y
    s = pd.Series(s.reshape(-1), name=args.sensitive)
    y = pd.DataFrame(y, columns=[args.target])

    X_tensor = torch.FloatTensor(X)
    X_tensor = X_tensor.to(device)
    model.eval()
    with torch.no_grad():
        z, _, _ = model(X_tensor)

    # array, series, DF
    return np.array(z.cpu()), s, y

def maxent_encode_yaleb(args, model, device, is_train=True, data=0):
    if data == 0:
        data = ExtendedYaleBDataLoader(args.data_path, args=args, device=device)
        data.load()

    if is_train:
        s = data.train_sensitive_label
        s = F.one_hot(s, num_classes=5)
        X = data.train_data
        y = data.train_label
    else:
        s = data.test_sensitive_label
        s = F.one_hot(s, num_classes=5)
        X = data.test_data
        y = data.test_label

    X_tensor, s_tensor, y_tensor = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(s)), torch.FloatTensor(np.array(y))
    X_tensor, s_tensor, y_tensor = X_tensor.to(device), s_tensor.to(device), y_tensor.to(device)
    model.eval()
    with torch.no_grad():
        z, _, _ = model(X_tensor)
    s = pd.Series(np.argmax(s, axis=1), name='sensitive')

    # array, series, tensor
    return np.array(z.cpu()), s, y