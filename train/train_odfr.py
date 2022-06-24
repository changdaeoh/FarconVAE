from loss import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR, LinearLR
from util.utils import CosineAnnealingWarmupRestarts
from model.odfr import ODFR
import time
import os
import wandb
import matplotlib.pyplot as plt
import collections


def train_odfr(args, train_loader, test_loader, model, device, model_file_name):
    # setup
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    lrs = []
    if args.scheduler == 'exp':
        scheduler = ExponentialLR(opt, args.exp_gamma)
    elif args.scheduler == 'lr':
        scheduler = LinearLR(opt, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
    elif args.scheduler == 'one':
        step_per_ep = (190//args.batch_size) if args.data_name == 'yaleb' else len(train_loader)
        scheduler = OneCycleLR(opt, max_lr=args.max_lr, steps_per_epoch=step_per_ep, epochs=args.epochs, anneal_strategy='linear', final_div_factor=args.fdf)
    elif args.scheduler == 'cos':
        scheduler = CosineAnnealingWarmupRestarts(opt, args.t_0, T_mult=args.t_mult, max_lr=args.max_lr, min_lr=args.min_lr, T_up=args.t_up, gamma=args.cos_gamma)
    else:
        scheduler = None

    if args.data_name == 'yaleb':
        Ls_loss = nn.CrossEntropyLoss()
    else:
        Ls_loss = nn.BCEWithLogitsLoss()

    criterion = Criterion(args, device).to(device)
    best_acc, patience, best_to = -1, 0, -1e5
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_tot_loss, ep_s_pred_loss, ep_t_pred_loss, ep_ent_loss, ep_od_loss, ep_t_correct, ep_s_correct, ep_tot_num = \
            0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0
        for X, s, y in train_loader:
            if args.data_name == 'yaleb':
                s = torch.argmax(s, dim=1)
            opt.zero_grad(set_to_none=True)

            if args.data_name != 'yaleb':
                X, s, y = X.to(device), s.to(device), y.to(device)
            output = model(X)  # (zt, zs), (t_mu, s_mu, t_lvar, s_lvar), (zs_s, zt_y, zt_s)
            s_zs, y_zt, _ = output[2]


            if args.encoder == 'mlp':
                L_s = Ls_loss(s_zs, s)
                for param in model.encoder.net.parameters():
                    param.requires_grad = False
                L_s.backward(retain_graph=True)
                for param in model.encoder.net.parameters():
                    param.requires_grad = True
            else:
                L_s = Ls_loss(s_zs, s)
                for param in model.encoder.net.parameters():
                    param.requires_grad = False
                L_s.backward(retain_graph=True)
                for param in model.encoder.net.parameters():
                    param.requires_grad = True
                

            # calculate tot loss except CE(s), update entire network
            tot_loss, L_t, ent_loss, od_loss = criterion(output, y, epoch)
            tot_loss.backward()
            opt.step()

            # update scheduler - iter wise
            if (args.scheduler == 'one') or (args.scheduler == 'cos'):
                scheduler.step()
                lrs.append(opt.param_groups[0]['lr'])

            # Train loss tracking
            ep_tot_num += X.shape[0]
            ep_s_pred_loss += L_s.item() * X.shape[0]
            ep_t_pred_loss += L_t.item() * X.shape[0]
            ep_ent_loss += ent_loss.item() * X.shape[0]
            ep_od_loss += od_loss.item() * X.shape[0]
            ep_tot_loss += (tot_loss.item() + L_s.item()) * X.shape[0]
            if args.data_name == 'yaleb':
                ep_t_correct += (torch.argmax(y_zt, dim=1) == y).sum().item()
                ep_s_correct += (torch.argmax(s_zs, dim=1) == s).sum().item()
            else:
                y_mask = torch.sigmoid(y_zt) < 0.5
                y_hat_bin = torch.ones_like(y_zt, device=device)
                y_hat_bin[y_mask] = 0.0
                ep_t_correct += (y_hat_bin == y).sum().item()

                s_mask = torch.sigmoid(s_zs) < 0.5
                s_hat_bin = torch.ones_like(s_zs, device=device)
                s_hat_bin[s_mask] = 0.0
                ep_s_correct += (s_hat_bin == s).sum().item()

        # update scheduler (epoch wise)
        if (args.scheduler == 'exp') or (args.scheduler == 'lr'):
            scheduler.step()
            lrs.append(opt.param_groups[0]['lr'])

        # ------------------------------------
        # validation loop per epoch
        # ------------------------------------
        model.eval()
        ep_tot_loss_test, ep_s_pred_loss_test, ep_t_pred_loss_test, ep_ent_loss_test, ep_od_loss_test, ep_t_correct_test, ep_s_correct_test, ep_tot_test_num = \
            0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for X, s, y in test_loader:
                if args.data_name == 'yaleb':
                    s = torch.argmax(s, dim=1)
                if args.data_name != 'yaleb':
                    X, s, y = X.to(device), s.to(device), y.to(device)
                output = model(X)
                s_zs, y_zt, _ = output[2]

                # get test loss
                L_s_test = Ls_loss(s_zs, s)
                tot_loss_test, L_t_test, ent_loss_test, od_loss_test = criterion(output, y, epoch)

                # Test loss tracking
                ep_tot_test_num += X.shape[0]
                if args.data_name == 'yaleb':
                    ep_t_correct_test += (torch.argmax(y_zt, dim=1) == y).sum().item()
                    ep_s_correct_test += (torch.argmax(s_zs, dim=1) == s).sum().item()
                else:
                    y_mask = torch.sigmoid(y_zt) < 0.5
                    y_hat_bin = torch.ones_like(y_zt, device=device)
                    y_hat_bin[y_mask] = 0.0
                    ep_t_correct_test += (y_hat_bin == y).sum().item()

                    s_mask = torch.sigmoid(s_zs) < 0.5
                    s_hat_bin = torch.ones_like(s_zs, device=device)
                    s_hat_bin[s_mask] = 0.0
                    ep_s_correct_test += (s_hat_bin == s).sum().item()
                ep_s_pred_loss_test += L_s_test.item() * X.shape[0]
                ep_t_pred_loss_test += L_t_test.item() * X.shape[0]
                ep_ent_loss_test += ent_loss_test.item() * X.shape[0]
                ep_od_loss_test += od_loss_test.item() * X.shape[0]
                ep_tot_loss_test += (tot_loss_test.item() + L_s_test.item()) * X.shape[0]

        # -------------------------------------
        # Monitoring entire process
        # -------------------------------------
        log_dict = collections.defaultdict(float)
        log_dict['s_pred_loss'] = (ep_s_pred_loss / ep_tot_num)
        log_dict['t_pred_loss'] = (ep_t_pred_loss / ep_tot_num)
        log_dict['ent_loss'] = (ep_ent_loss / ep_tot_num)
        log_dict['od_loss'] = (ep_od_loss / ep_tot_num)
        log_dict['y_acc_train'] = (ep_t_correct / ep_tot_num)
        log_dict['s_acc_train'] = (ep_s_correct / ep_tot_num)

        log_dict['s_pred_loss_test'] = (ep_s_pred_loss_test / ep_tot_test_num)
        log_dict['t_pred_loss_test'] = (ep_t_pred_loss_test / ep_tot_test_num)
        log_dict['ent_loss_test'] = (ep_ent_loss_test / ep_tot_test_num)
        log_dict['od_loss_test'] = (ep_od_loss_test / ep_tot_test_num)
        log_dict['y_acc_test'] = (ep_t_correct_test / ep_tot_test_num)
        log_dict['s_acc_test'] = (ep_s_correct_test / ep_tot_test_num)
        wandb.log(log_dict)

        if (epoch % 10) == 0:
            print('Epoch: [{}/{}]\nTrain - L_pred_s: {:.3f}, L_pred_y: {:.3f}, L_ent: {:.3f}, L_od: {:.3f}'
                '\n       y_acc: {:.3f}, s_acc: {:.3f}'
                '\nTest - L_pred_s: {:.3f}, L_pred_y: {:.3f}, L_ent: {:.3f}, L_od: {:.3f}'
                '\n       y_acc: {:.3f}, s_acc: {:.3f}'
                .format(epoch, args.epochs, ep_s_pred_loss / ep_tot_num, ep_t_pred_loss / ep_tot_num,
                        ep_ent_loss / ep_tot_num, ep_od_loss / ep_tot_num, ep_t_correct / ep_tot_num, ep_s_correct / ep_tot_num,
                        ep_t_pred_loss_test / ep_tot_test_num, ep_s_pred_loss_test / ep_tot_test_num, ep_ent_loss_test / ep_tot_test_num,
                        ep_od_loss_test / ep_tot_test_num, ep_t_correct_test / ep_tot_test_num, ep_s_correct_test / ep_tot_test_num))

        # save best model
        tt_acc = ep_t_correct_test / ep_tot_test_num
        current_to = tt_acc - ((ep_ent_loss_test + ep_od_loss_test)/(ep_tot_test_num))
        if args.save_criterion == 'yacc':
            if tt_acc > best_acc:
                best_acc = tt_acc
                torch.save(model.state_dict(),
                        os.path.join(args.model_path, f'{args.data_name}_odfr_{model_file_name}.pt'))
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
        else:
            if best_to < current_to:
                best_to = current_to
                torch.save(model.state_dict(), os.path.join(args.model_path, f'{args.data_name}_odfr_{model_file_name}.pt'))
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

    # plt.plot(lrs)
    # wandb.log({'train LR schedule_' + args.data_name : wandb.Image(plt)})
    # plt.close()

    # return best model
    if args.last_epmod :
        return model
    else:
        best_model = ODFR(args, device)
        best_model.load_state_dict(torch.load(args.model_path + f'/{args.data_name}_odfr_{model_file_name}.pt'))
        best_model.eval()
        best_model.to(device)
        return best_model