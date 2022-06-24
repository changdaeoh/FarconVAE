from loss import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR, LinearLR
from util.utils import CosineAnnealingWarmupRestarts
from model.maxent import MaxEnt
import collections
import os
import wandb
import matplotlib.pyplot as plt


def train_maxent(args, train_loader, test_loader, model, device, model_file_name):
    # optimizers and schedulers
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)
    opt_tar = torch.optim.Adam(model.target_net.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)
    opt_disc = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    # lrs = []
    if args.scheduler == 'exp':
        pass
    elif args.scheduler == 'lr':
        scheduler_e = LinearLR(opt_enc, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        scheduler_t = LinearLR(opt_tar, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        scheduler_d = LinearLR(opt_disc, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
    elif args.scheduler == 'one':
        step_per_ep = (190//args.batch_size) if args.data_name == 'yaleb' else len(train_loader)
        scheduler_e = OneCycleLR(opt_enc, max_lr=args.max_lr, steps_per_epoch=step_per_ep, epochs=args.epochs, anneal_strategy='linear', final_div_factor=args.fdf)
        scheduler_t = OneCycleLR(opt_tar, max_lr=args.max_lr, steps_per_epoch=step_per_ep, epochs=args.epochs, anneal_strategy='linear', final_div_factor=args.fdf)
        scheduler_d = OneCycleLR(opt_disc, max_lr=args.max_lr, steps_per_epoch=step_per_ep, epochs=args.epochs, anneal_strategy='linear', final_div_factor=args.fdf)
    elif args.scheduler == 'cos':
        scheduler_e = CosineAnnealingWarmupRestarts(opt_enc, args.t_0, T_mult=args.t_mult, max_lr=args.max_lr, min_lr=args.min_lr, T_up=args.t_up, gamma=args.cos_gamma)
        scheduler_t = CosineAnnealingWarmupRestarts(opt_tar, args.t_0, T_mult=args.t_mult, max_lr=args.max_lr, min_lr=args.min_lr, T_up=args.t_up, gamma=args.cos_gamma)
        scheduler_d = CosineAnnealingWarmupRestarts(opt_disc, args.t_0, T_mult=args.t_mult, max_lr=args.max_lr, min_lr=args.min_lr, T_up=args.t_up, gamma=args.cos_gamma)
    else:
        scheduler_e = None
        scheduler_t = None
        scheduler_d = None

    pred_loss = nn.CrossEntropyLoss() if args.data_name == 'yaleb' else nn.BCEWithLogitsLoss()
    
    best_acc, patience, best_to = -100000, 0, -100000
    for epoch in range(1, args.epochs + 1):
        ep_enctar_loss, ep_s_pred_loss, ep_t_pred_loss, ep_adv_loss, ep_t_correct, ep_s_correct, ep_tot_num = \
            0.0, 0.0, 0.0, 0.0, 0, 0, 0
        model.encoder.train()
        model.target_net.train()
        model.discriminator.train()
        for X, s, y in train_loader:
            n = X.shape[0]
            if args.data_name == 'yaleb':
                s = torch.argmax(s, dim=1)

            if args.data_name != 'yaleb':
                X, s, y = X.to(device), s.to(device), y.to(device)
            z, y_pred, s_pred = model(X)

            # target prediction loss
            t_pred_loss = pred_loss(y_pred, y)

            # entropy maximizing loss
            if args.model_name == 'maxent':
                adv_loss = entropy_loss(s_pred, agg=args.kld_agg, data_name=args.data_name)
            elif args.model_name == 'maxml':
                adv_loss = - pred_loss(s_pred, s)
            else:
                raise ValueError
            
            # loss weighted sum
            enctar_loss = t_pred_loss + args.alpha * adv_loss
            
            # param optimize
            opt_enc.zero_grad()
            opt_tar.zero_grad()
            enctar_loss.backward()
            opt_enc.step()
            opt_tar.step()

            # discriminator update
            z_disc = z.clone().detach()  # ! fix encoder. update only disc
            s_pred_disc = model.discriminator(z_disc)
            s_pred_loss = pred_loss(s_pred_disc, s)
            opt_disc.zero_grad()
            s_pred_loss.backward()
            opt_disc.step()
            
            if (args.scheduler == 'one') or (args.scheduler == 'cos'):
                scheduler_e.step()
                scheduler_t.step()
                scheduler_d.step()
                # lrs.append(opt.param_groups[0]['lr'])

            # Train loss tracking
            ep_tot_num += n
            ep_s_pred_loss += s_pred_loss.item() * n
            ep_t_pred_loss += t_pred_loss.item() * n
            ep_adv_loss += args.alpha * adv_loss.item() * n
            ep_enctar_loss += (ep_adv_loss + ep_t_pred_loss)

            if args.data_name == 'yaleb':
                ep_t_correct += (torch.argmax(y_pred, dim=1) == y).sum().item()
                ep_s_correct += (torch.argmax(s_pred_disc, dim=1) == s).sum().item()
            else:  # tabular
                y_mask = torch.sigmoid(y_pred) < 0.5
                y_hat_bin = torch.ones_like(y_pred, device=device)
                y_hat_bin[y_mask] = 0.0
                ep_t_correct += (y_hat_bin == y).sum().item()

                s_mask = torch.sigmoid(s_pred_disc) < 0.5
                s_hat_bin = torch.ones_like(s_pred_disc, device=device)
                s_hat_bin[s_mask] = 0.0
                ep_s_correct += (s_hat_bin == s).sum().item()

        # update scheduler (epoch wise)
        if (args.scheduler == 'exp') or (args.scheduler == 'lr'):
            scheduler_e.step()
            scheduler_t.step()
            scheduler_d.step()
            # lrs.append(opt.param_groups[0]['lr'])

        # ------------------------------------
        # validation loop per epoch
        # ------------------------------------
        model.eval()
        ep_enctar_loss_test, ep_s_pred_loss_test, ep_t_pred_loss_test, ep_adv_loss_test, ep_t_correct_test, ep_s_correct_test, ep_tot_test_num = \
            0.0, 0.0, 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for X, s, y in test_loader:
                n = X.shape[0]
                if args.data_name == 'yaleb':
                    s = torch.argmax(s, dim=1)
                if args.data_name != 'yaleb':
                    X, s, y = X.to(device), s.to(device), y.to(device)
                z, y_pred, s_pred = model(X)

                # target prediction loss
                t_pred_loss_test = pred_loss(y_pred, y)

                # entropy maximizing loss
                if args.model_name == "maxent":
                    adv_loss_test = entropy_loss(s_pred, agg=args.kld_agg, data_name=args.data_name)
                elif args.model_name == 'maxml':
                    adv_loss_test = - pred_loss(s_pred, s)
                else:
                    raise ValueError

                # loss weighted sum
                enctar_loss = t_pred_loss_test + args.alpha * adv_loss_test
       
                # discriminator update w.r.t sensitive prediction loss
                s_pred_disc = model.discriminator(z)
                s_pred_loss_test = pred_loss(s_pred_disc, s)

                # Test loss tracking
                ep_tot_test_num += n
                if args.data_name == 'yaleb':
                    ep_t_correct_test += (torch.argmax(y_pred, dim=1) == y).sum().item()
                    ep_s_correct_test += (torch.argmax(s_pred_disc, dim=1) == s).sum().item()
                else:
                    y_mask = torch.sigmoid(y_pred) < 0.5
                    y_hat_bin = torch.ones_like(y_pred, device=device)
                    y_hat_bin[y_mask] = 0.0
                    ep_t_correct_test += (y_hat_bin == y).sum().item()

                    s_mask = torch.sigmoid(s_pred_disc) < 0.5
                    s_hat_bin = torch.ones_like(s_pred_disc, device=device)
                    s_hat_bin[s_mask] = 0.0
                    ep_s_correct_test += (s_hat_bin == s).sum().item()
                ep_s_pred_loss_test += s_pred_loss_test.item() * n
                ep_t_pred_loss_test += t_pred_loss_test.item() * n
                ep_adv_loss_test += adv_loss_test.item() * n
                
                ep_enctar_loss_test += (ep_t_pred_loss_test + ep_adv_loss_test)

        # -------------------------------------
        # Monitoring entire process
        # -------------------------------------
        log_dict = collections.defaultdict(float)
        log_dict['s_pred_loss'] = (ep_s_pred_loss / ep_tot_num)
        log_dict['t_pred_loss'] = (ep_t_pred_loss / ep_tot_num)
        log_dict['adv_loss'] = (ep_adv_loss / ep_tot_num)
        log_dict['tot_enctar_loss'] = (ep_enctar_loss / ep_tot_num)
        log_dict['y_acc_train'] = (ep_t_correct / ep_tot_num)
        log_dict['s_acc_train'] = (ep_s_correct / ep_tot_num)

        log_dict['s_pred_loss_test'] = (ep_s_pred_loss_test / ep_tot_test_num)
        log_dict['t_pred_loss_test'] = (ep_t_pred_loss_test / ep_tot_test_num)
        log_dict['adv_loss_test'] = (ep_adv_loss_test / ep_tot_test_num)
        log_dict['tot_enctar_loss_Test'] = (ep_enctar_loss_test / ep_tot_test_num)
        log_dict['y_acc_test'] = (ep_t_correct_test / ep_tot_test_num)
        log_dict['s_acc_test'] = (ep_s_correct_test / ep_tot_test_num)
        wandb.log(log_dict)

        if (epoch % 10) == 0:
            print('Epoch: [{}/{}]\nTrain - L_pred_s: {:.3f}, L_pred_y: {:.3f}, L_adv: {:.3f} y_acc: {:.3f}, s_acc: {:.3f}'
                '\nTest - L_pred_s: {:.3f}, L_pred_y: {:.3f}, L_adv: {:.3f}, y_acc: {:.3f}, s_acc: {:.3f}'
                .format(epoch, args.epochs, ep_s_pred_loss / ep_tot_num, ep_t_pred_loss / ep_tot_num,
                        ep_adv_loss / ep_tot_num, ep_t_correct / ep_tot_num, ep_s_correct / ep_tot_num,
                        ep_t_pred_loss_test / ep_tot_test_num, ep_s_pred_loss_test / ep_tot_test_num, ep_adv_loss_test / ep_tot_test_num,
                        ep_t_correct_test / ep_tot_test_num, ep_s_correct_test / ep_tot_test_num))

        # save best model
        tt_acc = ep_t_correct_test / ep_tot_test_num
        current_to = tt_acc - (ep_enctar_loss_test /ep_tot_test_num) 
        if args.save_criterion == 'yacc':
            if tt_acc > best_acc:
                best_acc = tt_acc
                torch.save(model.state_dict(),
                        os.path.join(args.model_path, f'{args.data_name}_{args.model_name}_{model_file_name}'))
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
                torch.save(model.state_dict(), os.path.join(args.model_path, f'{args.data_name}_{args.model_name}_{model_file_name}'))
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
        best_model = MaxEnt(args, device)
        best_model.load_state_dict(torch.load(args.model_path + f'/{args.data_name}_{args.model_name}_{model_file_name}'))
        best_model.eval()
        best_model.to(device)
        return best_model