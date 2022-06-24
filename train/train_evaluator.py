from model.base import Predictor
from util.utils import *
import torch
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from model.eval_model import *
import collections
import os
# import wandb


def train_yaleb_evaluator(args, X, y, X_te, y_te, model, device, flag='Fair', eval_model='mlp'):
    print("==============================================")
    print(f"2stage training start.\n dataset size: {X.shape}")
    print("==============================================")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=1)
    loss_f = nn.CrossEntropyLoss()
    
    if args.scheduler == 'lr':
        scheduler = LinearLR(opt, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
    elif args.scheduler == 'one':
        scheduler = OneCycleLR(opt, max_lr=args.max_lr, steps_per_epoch=190//args.batch_size, epochs=args.epochs, anneal_strategy='linear')
    else:
        scheduler = None

    best_acc, patience = -1, 0
    for epoch in range(1, args.epochs + 1):
        # ------------------------------------
        # training
        # ------------------------------------
        model.train()
        ep_loss_train, ep_train_correct, ep_tot_num = 0.0, 0, 0
        opt.zero_grad()
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_f(pred, y)
        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()

        ep_tot_num += y.shape[0]
        ep_loss_train += loss.item() * y.shape[0]
        ep_train_correct += (torch.argmax(pred, dim=1) == y).sum()

        # ------------------------------------
        # evaluation
        # ------------------------------------
        model.eval()
        ep_loss_test, ep_test_correct, ep_tot_test_num = 0.0, 0, 0
        with torch.no_grad():
            X_te, y_te = X_te.to(device), y_te.to(device)
            pred = model(X_te)
            loss_test = loss_f(pred, y_te)

            # monitoring
            ep_tot_test_num += y_te.shape[0]
            ep_loss_test += loss_test.item() * y_te.shape[0]
            pred_int = torch.argmax(pred, dim=1)
            ep_test_correct += (pred_int == y_te).sum()


        test_acc = ep_test_correct / ep_tot_test_num

        # d = collections.defaultdict(float)
        # d[f'2stage_{eval_model}_{flag}_train_loss'] = loss.item()
        # d[f'2stage_{eval_model}_{flag}_train_acc'] = ep_train_correct / ep_tot_num
        # d[f'2stage_{eval_model}_{flag}_test_acc'] = test_acc
        # wandb.log(d)

        # save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.result_path + f'/{args.data_name}_downstream_{flag}_{args.down_save_name}.pt'))
        else:
            if args.early_stop:
                patience += 1
                if (patience % 10) == 0 :
                    print(f"----------------------------------- increase patience {patience}/{args.patience}")
                if patience > args.patience:
                    print(f"----------------------------------- early stopping")
                    break

        if (epoch % 10) == 0:
            print(flag + ' CLF - Ep: [{}/{}] Train Loss: {:.3f}, Train Accuracy: {:.3f}, Test Loss: {:.3f}, Test Accuracy: {:.3f}'
            .format(epoch, args.epochs, ep_loss_train / ep_tot_num, ep_train_correct / ep_tot_num, ep_loss_test / ep_tot_test_num, ep_test_correct / ep_tot_test_num))

    if flag == 's':
        if args.last_epmod_eval:
            return model
        else:
            model.load_state_dict(torch.load(args.result_path + f'/{args.data_name}_downstream_{flag}_{args.down_save_name}.pt'))
            model.eval()    
            return model
    else:
        model.load_state_dict(torch.load(args.result_path + f'/{args.data_name}_downstream_{flag}_{args.down_save_name}.pt'))
        model.eval()    
        return model


def train_evalutator(args, target, train_dl, test_dl, device, eval_model):
    print(f'eval tot iters :{len(train_dl) * args.epochs}')
    print("==============================================")
    print(f"2stage training start.\n dataset size: {train_dl.dataset.z.shape}")
    print("==============================================")
    if target == 'y':
        if eval_model == 'lr':
            predictor = OneLinearLayer(args.latent_dim, args.y_dim)
        elif eval_model == 'mlp':
            predictor = Predictor(args.latent_dim, args.y_dim, args.hidden_units, args)
    else:
        predictor = Predictor(args.latent_dim, args.s_dim, args.hidden_units, args=args)
    predictor = predictor.to(device)

    opt = torch.optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=1)
    loss_func = nn.BCEWithLogitsLoss() if args.data_name != 'yaleb' else nn.CrossEntropyLoss()

    if args.scheduler == 'lr':
        scheduler = LinearLR(opt, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
    elif args.scheduler == 'one':
        scheduler = OneCycleLR(opt, max_lr=args.max_lr, steps_per_epoch=len(train_dl), epochs=args.epochs, anneal_strategy='linear')
    else:
        scheduler = None

    best_acc = -1
    patience = 0; train_correct = 0; n_train = 0
    for epoch in range(1, args.epochs + 1):
        # train 
        predictor.train()
        for feature, s, y in train_dl:
            n_train += feature.shape[0]
            feature, s, y = feature.to(device), s.to(device), y.to(device)
            prediction_logit = predictor(feature)
            if target == 'y':
                loss = loss_func(prediction_logit, y)
            else:
                loss = loss_func(prediction_logit, s)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_correct += ((prediction_logit > 0) == s).sum().item() 
            if args.scheduler == 'one':
                scheduler.step()
        if args.scheduler == 'lr':
            scheduler.step()

        train_acc = train_correct / n_train

        # evaluation
        predictor.eval()
        N_test, correct_cnt, ep_loss_test, prediction_cnt = 0, 0, 0.0, 0
        with torch.no_grad():
            for feature, s, y in test_dl:
                feature, s, y = feature.to(device), s.to(device), y.to(device)
                N_test += feature.size(0)

                prediction_logit = predictor(feature)
                if target == 'y':
                    loss_test = loss_func(prediction_logit, y)
                    correct_cnt += ((prediction_logit > 0) == y).sum().item()
                else:
                    loss_test = loss_func(prediction_logit, s)
                    correct_cnt += ((prediction_logit > 0) == s).sum().item()
                prediction_cnt += (prediction_logit > 0).sum().item()
                ep_loss_test += loss_test.item() * feature.size(0)

        # ep performance
        test_loss, test_acc = ep_loss_test / N_test, correct_cnt / N_test

        # d = collections.defaultdict(float)
        # d[f'2stage_{eval_model}_{target}_train_loss'] = loss.item()
        # d[f'2stage_{eval_model}_{target}_train_acc'] = train_acc
        # d[f'2stage_{eval_model}_{target}_test_acc'] = test_acc
        # wandb.log(d)

        if (epoch % 5) == 0:
            print(f"-----------------ep:{epoch}/{args.epochs}")
            print(f"train acc: {train_acc:.3f}, test acc: {test_acc:.3f}\n")

        # save best model w.r.t. ep test acc
        best_predictor_path = args.result_path + f'/{args.data_name}_downstream_{target}_{args.down_save_name}.pt'
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(predictor.state_dict(), best_predictor_path)
        else:
            if args.early_stop:
                patience += 1
                if (patience % 10) == 0 :
                    print(f"----------------------------------- increase patience {patience}/{args.patience}")
                if patience > args.patience:
                    print(f"----------------------------------- early stopping")
                    break

    if target == 's':
        if args.last_epmod_eval:
            return predictor
        else:    
            predictor.load_state_dict(torch.load(best_predictor_path))
            return predictor
    else:
        predictor.load_state_dict(torch.load(best_predictor_path))
        return predictor