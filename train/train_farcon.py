from loss import *
from util.utils import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from model.base import BestClf
from model.farcon import FarconVAE
import torch.nn.functional as F

import os
import collections
# import wandb
# import pdb

#! ----------------------------- for tabular
def train_farconvae(args, model, train_loader, test_loader, device, model_file_name):
    ep_per_iter = len(train_loader)
    tot_train_iters = ep_per_iter * args.epochs
    print("==============================================")
    print(f"1stage training start.\n dataset size: {train_loader.dataset.X.shape}")
    print("==============================================")
    print(f'len train loader :{ep_per_iter}')
    print(f'train tot iters :{tot_train_iters}')
    current_iter = 0

    # model / optimizer setup
    clf_xy = BestClf(args.n_features, args.y_dim, args.clf_hidden_units, args)
    if args.clf_path != 'no':
        clf_xy.load_state_dict(torch.load(args.clf_path, map_location=device))
    clf_xy = clf_xy.to(device)
    opt_clf = torch.optim.Adam(clf_xy.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    opt_vae, opt_pred = model.get_optimizer()
    
    lrs = []
    if args.scheduler == 'lr':
        scheduler_c = LinearLR(opt_clf, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        scheduler_v = LinearLR(opt_vae, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        scheduler_d = LinearLR(opt_pred, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        schedulers = [scheduler_c, scheduler_v, scheduler_d]
    elif args.scheduler == 'one':
        scheduler_c = OneCycleLR(opt_clf, max_lr=args.max_lr, steps_per_epoch=ep_per_iter, epochs=args.epochs, anneal_strategy='linear')
        scheduler_v = OneCycleLR(opt_vae, max_lr=args.max_lr, steps_per_epoch=ep_per_iter, epochs=args.epochs, anneal_strategy='linear')
        scheduler_d = OneCycleLR(opt_pred, max_lr=args.max_lr, steps_per_epoch=ep_per_iter, epochs=args.epochs, anneal_strategy='linear')
        schedulers = [scheduler_c, scheduler_v, scheduler_d]
    else:
        schedulers = []

    clf_lossf = nn.BCEWithLogitsLoss()
    farcon_lossf = FarconVAELoss(args, device, tot_train_iters)

    best_clf_acc, best_pred_acc, best_to, patience = -1e7, -1e7, -1e7, 0
    for epoch in range(1, args.epochs + 1):
        # ------------------------------------
        # Training
        # ------------------------------------
        model.train()
        clf_xy.train() if args.clf_path == 'no' else clf_xy.eval()

        # tracking quantities (loss, variance, divergence, performance, ...)
        ep_tot_loss, ep_recon_loss, ep_kl_loss, ep_c_loss, ep_sr, \
        ep_pred_loss, ep_clf_loss, ep_pred_cor, ep_clf_cor, ep_tot_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

        for x, s, y in train_loader:
            x, s, y = x.to(device), s.to(device), y.to(device)

            current_iter += 1
            n = x.shape[0]
            ori_x, ori_s, ori_y = x, s, y
            cont_x, cont_s, cont_y = x, 1-s, y
            
            # BestClf loss
            y_hat = clf_xy(ori_x)
            clf_loss = clf_lossf(y_hat, ori_y)
            y_hat_bin = torch.ones_like(y_hat, device=device)
            y_hat_bin[y_hat < 0] = 0.0

            # out : (zx, zs), (x_recon, s_recon), (mu_x, logvar_x, mu_s, logvar_s), y_pred
            out1 = model(ori_x, ori_s, ori_y)
            out2 = model(cont_x, cont_s, cont_y)

            recon_loss, kl_loss, pred_loss, cont_loss, sr_loss, y_pred = farcon_lossf(out1, out2, ori_x, cont_x, ori_s, cont_s, y, model, current_iter)

            opt_vae.zero_grad(set_to_none=True)
            opt_pred.zero_grad(set_to_none=True)
            if args.clf_path == 'no':
                opt_clf.zero_grad(set_to_none=True)

            loss = recon_loss + pred_loss + kl_loss + cont_loss + sr_loss
            loss.backward()
            if args.clf_path == 'no':
                clf_loss.backward()

            if args.clip_val != 0:
                torch.nn.utils.clip_grad_norm_(model.vae_params(), max_norm=args.clip_val)
            opt_vae.step()
            opt_pred.step()
            if args.clf_path == 'no':
                opt_clf.step()

            if (args.scheduler == 'one'):
                for sche in schedulers:
                    sche.step()

            # monitoring
            ep_tot_num += n
            ep_tot_loss += loss.item() * n
            ep_recon_loss += recon_loss.item() * n
            ep_kl_loss += kl_loss.item() * n
            ep_pred_loss += pred_loss.item() * n
            ep_pred_cor += (y_pred == ori_y).sum().item()
            ep_clf_loss += clf_loss.item() * n
            ep_clf_cor += (y_hat_bin == ori_y).sum().item()
            if args.alpha != 0:
                ep_c_loss += cont_loss.item() * n
            if args.gamma != 0:
                ep_sr += sr_loss.item() * n

        if args.scheduler == 'lr':
            for sche in schedulers:
                sche.step()
            lrs.append(opt_vae.param_groups[0]['lr'])

        # ------------------------------------
        # validation
        # ------------------------------------
        model.eval()
        ep_recon_x_loss_test, ep_recon_s_loss_test, ep_pred_loss_test, ep_clf_loss_test = 0.0, 0.0, 0.0, 0.0
        ep_tot_test_num, ep_pred_cor_test, ep_clf_cor_test, ep_c_loss_test, current_to = 0, 0, 0, 0.0, 0.0
        y_pred_raw = torch.tensor([], device=device)
        with torch.no_grad():
            for x, s, y in test_loader:
                x, s, y = x.to(device), s.to(device), y.to(device)
                n = x.shape[0]
                clf_loss_test = clf_lossf(clf_xy(x), y)
                # use binary for pseudo label in test time encoder input y
                y_hat = clf_xy(x)
                y_hat_bin = torch.ones_like(y_hat, device=device)
                y_hat_bin[y_hat < 0] = 0.0

                out1 = model(x, s, y_hat_bin)
                out2 = model(x, 1-s, y_hat_bin)

                recon_x_loss_te, recon_s_loss_te, pred_loss_te, cont_loss_te, y_pred_te = farcon_lossf(out1, out2, x, x, s, 1-s, y, model, current_iter, is_train=False)
                
                ep_tot_test_num += n
                y_pred_raw = torch.cat((y_pred_raw, y_pred_te))
                ep_recon_x_loss_test += recon_x_loss_te.item() * n
                ep_recon_s_loss_test += recon_s_loss_te.item() * n
                ep_c_loss_test += args.alpha * cont_loss_te.item() * n
                ep_pred_loss_test += pred_loss_te.item() * n
                ep_pred_cor_test += (y_pred_te == y).sum().item()
                ep_clf_cor_test += (y_hat_bin == y).sum().item()
                ep_clf_loss_test += clf_loss_test.item() * n

        # save best CLF w.r.t epoch best accuracy
        if args.clf_path == 'no':
            if best_clf_acc < ep_clf_cor_test:
                best_clf_acc = ep_clf_cor_test
                torch.save(clf_xy.state_dict(), os.path.join(args.model_path, 'clf_' + model_file_name))

        # save farcon w.r.t epoch best trade off
        ep_pred_test_acc = ep_pred_cor_test /ep_tot_test_num
        current_to = ep_pred_test_acc - (ep_c_loss_test / ep_tot_test_num)
        if best_to < current_to:
            best_to = current_to
            best_pred_acc = ep_pred_test_acc
            torch.save(model.state_dict(), os.path.join(args.model_path, 'farcon_' + model_file_name))
            patience = 0
        else:
            if args.early_stop:
                patience += 1
                if (patience % 10) == 0 :
                    print(f"----------------------------------- increase patience {patience}/{args.patience}")
                if patience > args.patience:
                    print(f"----------------------------------- early stopping")
                    break
            else:
                pass

        # -------------------------------------
        # Monitoring entire process
        # -------------------------------------
        # log_dict = collections.defaultdict(float)
        # log_dict['loss'] = ep_tot_loss / ep_tot_num
        # log_dict['recon_loss'] = (ep_recon_loss / ep_tot_num)
        # log_dict['recon_s_loss_test'] = (ep_recon_s_loss_test / ep_tot_test_num)
        # log_dict['recon_x_loss_test'] = (ep_recon_x_loss_test / ep_tot_test_num)
        # log_dict['kl_loss'] = (ep_kl_loss / ep_tot_num)
        # log_dict['c_loss'] = (ep_c_loss / ep_tot_num)
        # log_dict['pred_loss'] = (ep_pred_loss / ep_tot_num)
        # log_dict['clf_loss'] = (ep_clf_loss / ep_tot_num)
        # log_dict['pred_acc'] = (ep_pred_cor / ep_tot_num)
        # log_dict['clf_acc'] = (ep_clf_cor / ep_tot_num)
        # log_dict['ms_loss'] = (ep_sr / ep_tot_num)
        # log_dict['pred_loss_test'] = (ep_pred_loss_test / ep_tot_test_num)
        # log_dict['clf_loss_test'] = (ep_clf_loss_test / ep_tot_test_num)
        # log_dict['pred_acc_test'] = (ep_pred_cor_test / ep_tot_test_num)
        # log_dict['clf_acc_test'] = (ep_clf_cor_test / ep_tot_test_num)
        # wandb.log(log_dict)

        if (epoch % 10) == 0:
            print(f'\nEp: [{epoch}/{args.epochs}] (TRAIN) ---------------------\n'
                f'Loss: {ep_tot_loss / ep_tot_num:.3f}, L_rx: {ep_recon_loss / ep_tot_num:.3f}, '
                f'L_kl: {ep_kl_loss / ep_tot_num:.3f}, L_c: {ep_c_loss / ep_tot_num:.3f}, \n'
                f'L_clf: {ep_clf_loss / ep_tot_num:.3f}, L_pred: {ep_pred_loss / ep_tot_num:.3f}, '
                f'L_clf_acc: {ep_clf_cor / ep_tot_num:.3f}, L_pred_acc: {ep_pred_cor / ep_tot_num:.3f}')
            print(f'Ep: [{epoch}/{args.epochs}] (VALID)  ---------------------\n'
                f'L_rx: {ep_recon_x_loss_test / ep_tot_test_num:.3f}, L_rs: {ep_recon_s_loss_test / ep_tot_test_num:.3f}, '
                f'L_pred: {ep_pred_loss_test / ep_tot_test_num:.3f}, L_clf: {ep_clf_loss_test / ep_tot_test_num:.3f}, \n'
                f'L_clf_acc: {ep_clf_cor_test / ep_tot_test_num:.3f}, L_pred_acc: {ep_pred_cor_test / ep_tot_test_num:.3f}')

    # log_dict['best_pred_test_acc'] = best_pred_acc

    # return best model !
    if args.clf_path == 'no':
        clf_xy = BestClf(args.n_features, args.y_dim, args.clf_hidden_units, args)
        clf_xy.load_state_dict(torch.load(os.path.join(args.model_path, 'clf_' + model_file_name), map_location=device))
        clf_xy.to(device)

    if args.last_epmod :
        return model, clf_xy
    else:
        model = FarconVAE(args, device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'farcon_' + model_file_name), map_location=device))
        model.to(device)
        return model, clf_xy

#! ------------------------ yaleB train function
def train_farconvae_yaleb(args, model, train_loader, test_loader, device, model_file_name):
    current_iter = 0
    tot_train_iters = (190 // args.batch_size) * args.epochs
    print(f"total train steps : {tot_train_iters}")

    # model / optimizer setup
    clf_xy = BestClf(args.n_features, args.y_dim, args.clf_hidden_units, args)
    if args.clf_path != 'no':
        clf_xy.load_state_dict(torch.load(args.clf_path, map_location=device))
    clf_xy = clf_xy.to(device)
    opt_clf = torch.optim.Adam(clf_xy.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    opt_vae, opt_pred = model.get_optimizer()

    lrs = []
    if args.scheduler == 'lr':
        scheduler_c = LinearLR(opt_clf, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        scheduler_v = LinearLR(opt_vae, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        scheduler_d = LinearLR(opt_pred, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        schedulers = [scheduler_c, scheduler_v, scheduler_d]
        pass
    elif args.scheduler == 'one':
        scheduler_c = OneCycleLR(opt_clf, max_lr=args.max_lr, steps_per_epoch=190 // args.batch_size, epochs=args.epochs, anneal_strategy='linear')
        scheduler_v = OneCycleLR(opt_vae, max_lr=args.max_lr, steps_per_epoch=190 // args.batch_size, epochs=args.epochs, anneal_strategy='linear')
        scheduler_d = OneCycleLR(opt_pred, max_lr=args.max_lr, steps_per_epoch=190 // args.batch_size, epochs=args.epochs, anneal_strategy='linear')
        schedulers = [scheduler_c, scheduler_v, scheduler_d]
    else:
        schedulers = []
    
    clf_lossf = nn.CrossEntropyLoss()  # X -> y unfair predictor loss
    farcon_lossf = FarconVAELoss(args, device, tot_train_iters)

    for x, s, y in test_loader:
        s = torch.argmax(s, dim=1)
        x = x.cpu().numpy(); s = s.cpu().numpy(); y = y.cpu().numpy()
        # make test set contrastive pair
        oriset, contset = make_yaleb_test_contset(x, s, y)
        oriset = torch.from_numpy(oriset).to(device)
        contset = torch.from_numpy(contset).to(device)
        ori_x, ori_s, ori_y = oriset[:, :-2], oriset[:, -2], oriset[:, -1]
        cont_x, cont_s = contset[:, :-2], contset[:, -2]

    best_clf_acc, best_pred_acc, best_to, patience = -1e7, -1e7, -1e7, 0
    for epoch in range(1, args.epochs + 1):
        # ------------------------------------
        # Training
        # ------------------------------------
        model.train()
        clf_xy.train() if args.clf_path == 'no' else clf_xy.eval()
        ep_tot_loss, ep_recon_loss, ep_kl_loss, ep_c_loss, ep_sr, \
        ep_pred_loss, ep_clf_loss, ep_pred_cor, ep_clf_cor, ep_tot_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0

        for x_ori, x_cont, s_ori, s_cont, y in train_loader:
            current_iter += 1
            n = x_ori.shape[0]

            # out : (zx, zs), (x_recon, s_recon), (mu_x, logvar_x, mu_s, logvar_s), y_pred
            out1 = model(x_ori, s_ori, y, input_label=True)
            out2 = model(x_cont, s_cont, y, input_label=True)

            # BestClf loss
            y_hat = clf_xy(x_ori)
            y_hat_clf_label = torch.argmax(y_hat, dim=1)
            clf_loss = clf_lossf(y_hat, y)

            # FarconVAE loss
            recon_loss, kl_loss, pred_loss, cont_loss, sr_loss, y_pred = farcon_lossf(out1, out2, x_ori, x_cont, s_ori, s_cont, y, model, current_iter)
            loss = recon_loss + kl_loss + cont_loss + pred_loss + sr_loss

            opt_vae.zero_grad(set_to_none=True)
            opt_pred.zero_grad(set_to_none=True)
            if args.clf_path == 'no':
                opt_clf.zero_grad(set_to_none=True)

            loss.backward()
            if args.clf_path == 'no':
                clf_loss.backward()

            # gradient clipping for vae backbone
            if args.clip_val != 0:
                torch.nn.utils.clip_grad_norm_(model.vae_params(), max_norm=args.clip_val)
            opt_vae.step()
            opt_pred.step()
            if args.clf_path == 'no':
                opt_clf.step()

            # update scheduler (iter wise)
            if args.scheduler == 'one':
                for sche in schedulers:
                    sche.step()

            # Train loss tracking
            ep_tot_num += n
            ep_recon_loss += recon_loss.item() * n
            ep_kl_loss += kl_loss.item() * n
            ep_pred_loss += pred_loss.item() * n
            ep_pred_cor += (y_pred == y).sum().item()
            ep_clf_loss += clf_loss.item() * n
            ep_clf_cor += (y_hat_clf_label == y).sum().item()
            ep_tot_loss += loss.item() * n
            if args.alpha != 0:
                ep_c_loss += cont_loss.item() * n
            if args.gamma != 0:
                ep_sr += sr_loss.item() * n

        if args.scheduler == 'lr':
            for sche in schedulers:
                sche.step()
            lrs.append(opt_vae.param_groups[0]['lr'])

        # ------------------------------------
        # validation
        # ------------------------------------
        model.eval()
        ep_recon_x_loss_test, ep_recon_s_loss_test, ep_pred_loss_test, ep_clf_loss_test = 0.0, 0.0, 0.0, 0.0
        ep_tot_test_num, ep_pred_cor_test, ep_clf_cor_test, ep_c_loss_test = 0, 0, 0, 0.0
        current_to = 0.0

        ori_x, ori_s, ori_y = oriset[:, :-2], oriset[:, -2], oriset[:, -1]
        cont_x, cont_s = contset[:, :-2], contset[:, -2]
        y_pred_raw = torch.tensor([], device=device)
        with torch.no_grad():
            n = ori_x.shape[0]
            ori_x = ori_x.float(); cont_x = cont_x.float(); ori_y = ori_y.long()

            # clf loss
            clf_test_pred = clf_xy(ori_x)
            clf_loss_test = clf_lossf(clf_test_pred, ori_y)

            # forward
            y_hat_clf_test = torch.argmax(clf_test_pred, dim=1)
            y_hat_cont_clf_test = torch.argmax(clf_xy(cont_x), dim=1)
            ori_s = F.one_hot(ori_s.to(torch.int64), num_classes=5)
            cont_s = F.one_hot(cont_s.to(torch.int64), num_classes=5)

            out1 = model(ori_x, ori_s, y_hat_clf_test, input_label=True)
            out2 = model(cont_x, cont_s, y_hat_cont_clf_test, input_label=True)

            recon_x_loss_te, recon_s_loss_te, pred_loss_te, cont_loss_te, y_pred_te = farcon_lossf(out1, out2, ori_x, cont_x, ori_s, cont_s, ori_y, model, current_iter, is_train=False)
                
            ep_tot_test_num += n
            y_pred_raw = torch.cat((y_pred_raw, y_pred_te))
            ep_recon_x_loss_test += recon_x_loss_te.item() * n
            ep_recon_s_loss_test += recon_s_loss_te.item() * n
            ep_c_loss_test += cont_loss_te.item() * n
            ep_pred_loss_test += pred_loss_te.item() * n
            ep_pred_cor_test += (y_pred_te == ori_y).sum().item()
            ep_clf_cor_test += (y_hat_clf_test == ori_y).sum().item()
            ep_clf_loss_test += clf_loss_test.item() * n

        # --------------------------
        # tracking some importance metric
        # --------------------------
        # save best CLF w.r.t epoch best accuracy
        if args.clf_path == 'no':
            if best_clf_acc < ep_clf_cor_test:
                best_clf_acc = ep_clf_cor_test
                torch.save(clf_xy.state_dict(), os.path.join(args.model_path, 'clf_' + model_file_name))

        # save farcon w.r.t epoch best trade off
        ep_pred_test_acc = ep_pred_cor_test / ep_tot_test_num
        current_to = ep_pred_test_acc - (ep_c_loss_test / ep_tot_test_num)
        if best_to < current_to:
            best_to = current_to
            best_pred_acc = ep_pred_test_acc
            torch.save(model.state_dict(), os.path.join(args.model_path, 'farcon_' + model_file_name))
            patience = 0
        else:
            if args.early_stop:
                patience += 1
                if (patience % 10) == 0 :
                    print(f"----------------------------------- increase patience {patience}/{args.patience}")
                if patience > args.patience:
                    print(f"----------------------------------- early stopping")
                    break
            else:
                pass

        # log_dict = collections.defaultdict(float)
        # log_dict['loss'] = ep_tot_loss / ep_tot_num
        # log_dict['recon_loss'] = (ep_recon_loss / ep_tot_num)
        # log_dict['recon_s_loss_test'] = (ep_recon_s_loss_test / ep_tot_test_num)
        # log_dict['recon_x_loss_test'] = (ep_recon_x_loss_test / ep_tot_test_num)
        # log_dict['kl_loss'] = (ep_kl_loss / ep_tot_num)
        # log_dict['c_loss'] = (ep_c_loss / ep_tot_num)
        # log_dict['sr_loss'] = (ep_sr / ep_tot_num)
        # log_dict['pred_loss'] = (ep_pred_loss / ep_tot_num)
        # log_dict['clf_loss'] = (ep_clf_loss / ep_tot_num)
        # log_dict['pred_acc'] = (ep_pred_cor / ep_tot_num)
        # log_dict['clf_acc'] = (ep_clf_cor / ep_tot_num)
        # log_dict['pred_loss_test'] = (ep_pred_loss_test / ep_tot_test_num)
        # log_dict['clf_loss_test'] = (ep_clf_loss_test / ep_tot_test_num)
        # log_dict['pred_acc_test'] = (ep_pred_cor_test / ep_tot_test_num)
        # log_dict['clf_acc_test'] = (ep_clf_cor_test / ep_tot_test_num)
        # log_dict['c_loss_test'] = (ep_c_loss_test / ep_tot_test_num)

        # wandb.log(log_dict)

        if (epoch % 10) == 0:
            print(f'\nEp: [{epoch}/{args.epochs}] (TRAIN) ---------------------\n'
                f'Loss: {ep_tot_loss / ep_tot_num:.3f}, L_rec: {ep_recon_loss / ep_tot_num:.3f}, '
                f'L_kl: {ep_kl_loss / ep_tot_num:.3f}, L_c: {ep_c_loss / ep_tot_num:.3f}, \n'
                f'L_clf: {ep_clf_loss / ep_tot_num:.3f}, L_pred: {ep_pred_loss / ep_tot_num:.3f}, '
                f'L_clf_acc: {ep_clf_cor / ep_tot_num:.3f}, L_pred_acc: {ep_pred_cor / ep_tot_num:.3f}')
            print(f'Ep: [{epoch}/{args.epochs}] (VALID)  ---------------------\n'
                f'L_rx: {ep_recon_x_loss_test / ep_tot_test_num:.3f}, L_rs: {ep_recon_s_loss_test / ep_tot_test_num:.3f}, '
                f'L_clf: {ep_clf_loss_test / ep_tot_test_num:.3f}, L_pred: {ep_pred_loss_test / ep_tot_test_num:.3f}, \n'
                f'L_clf_acc: {ep_clf_cor_test / ep_tot_test_num:.3f}, L_pred_acc: {ep_pred_cor_test / ep_tot_test_num:.3f}')

    # log_dict['best_pred_test_acc'] = best_pred_acc

    # return bestclf
    if args.clf_path == 'no':
        clf_xy = BestClf(args.n_features, args.y_dim, args.clf_hidden_units, args)
        clf_xy.load_state_dict(torch.load(os.path.join(args.model_path, 'clf_' + model_file_name), map_location=device))
        clf_xy.to(device)

    # return best vs last model
    if args.last_epmod :
        return model, clf_xy
    else:
        model = FarconVAE(args, device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'farcon_' + model_file_name), map_location=device))
        model.to(device)
        return model, clf_xy