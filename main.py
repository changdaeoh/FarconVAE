from random import choice
from train.train_farcon import *
from model.farcon import *
from eval import *

import torch
import datetime
import time


def main(args, device):
    start_time = time.time()
    print(f'using {device}')
    seed_everything(args.seed)
    RUN_ID = str(datetime.datetime.now()).replace(' ','_').replace('-','_').replace('.',':').replace(':','_')

    # saving file names define
    if args.save_name == 'default':
        model_file_name = RUN_ID + '_' + args.model_name + '.pth'
        z_vis_name = RUN_ID + '_' + args.model_name + '_vis' + '.png'
    else:
        model_file_name = args.save_name + '.pth'
        z_vis_name = args.save_name + '_vis' + '.png'
    args.down_save_name = model_file_name
    # id_log = {'model ID': RUN_ID}
    # wandb.log(id_log)

    
    data = None
    if args.data_name == 'yaleb':
        data = ExtendedYaleBDataLoader(args.data_path, args=args, device=device)
        data.load()
        train_loader = torch.utils.data.DataLoader(data.pretrainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(data.testset, batch_size=args.batch_size_te, shuffle=False, num_workers=0)
    else:
        train_loader, test_loader = get_xsy_loaders(os.path.join(args.data_path, args.train_file_name),
                                                    os.path.join(args.data_path, args.test_file_name),
                                                    args.data_name, args.sensitive, args.batch_size_te, args)

    model = FarconVAE(args, device)
    model.to(device)
    
    # train model
    if args.data_name == 'yaleb':       model, clf_xy = train_farconvae_yaleb(args, model, train_loader, test_loader, device, model_file_name)
    else:                               model, clf_xy = train_farconvae(args, model, train_loader, test_loader, device, model_file_name)

    # Visualize learned representation
    if args.vis == 1:
        if args.data_name != 'yaleb':
            zx_te, zs_te, s_te, y_te = encode_all(args, test_loader.dataset, model, clf_xy, device, is_train=False)
            vis_z_binary(args, zs_te, zx_te, s_te, y_te, z_vis_name, is_train=False)
        else:
            zx_te, zs_te, s_te, y_te = encode_all_yaleb(args, model, clf_xy, device, is_train=False, data=data)
            vis_z_yaleb(args, zs_te, zx_te, s_te, y_te, z_vis_name, is_train=False)
        print("Time for Train & Vis :", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    else:
        print("Time for Train :", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if args.data_name == 'yaleb': return model, clf_xy, data
    else:                         return model, clf_xy, train_loader.dataset, test_loader.dataset


def evaluation(args, device, eval_model, model=None, clf_xy=None, is_e2e=True, trainset=None, testset=None):
    seed_everything(args.seed)
    print(f'using {device}')

    # Validate
    if is_e2e:
        print('do End-to-End Experiment phase')
        y_pred, y_acc, s_acc, s_pred = z_evaluator(args, model, clf_xy, device, eval_model, 'ours', trainset, testset)
    else:
        model = FarconVAE(args, device)
        clf_xy = BestClf(args.n_features, 1, args.hidden_units, args)
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_file), map_location=device))
        clf_xy.load_state_dict(torch.load(os.path.join(args.model_path, 'clf_'+args.model_file), map_location=device))
        model, clf_xy = model.to(device), clf_xy.to(device)

        # predict Y, S from learned representation using eval_model(LR or MLP) (s is always evaluated with MLP according to previous works)
        y_pred, y_acc, s_acc, s_pred = z_evaluator(args, model, clf_xy, device, eval_model, 'ours', trainset, testset)

    # Final reporting
    if args.data_name != 'yaleb':
        dp, _, _, eodd, gap = metric_scores(os.path.join(args.data_path, args.test_file_name), y_pred)
        print('----------------------------------------------------------------')
        print(f'DP: {dp:.4f} | EO: {eodd:.4f} | GAP: {gap:.4f} | yAcc: {y_acc:.4f} | sAcc: {s_acc:.4f}')
        print('----------------------------------------------------------------')
        #performance_log = {f'DP': dp, f'EO':eodd, f'GAP': gap, f'y_acc': y_acc, f's_acc': s_acc}
        #wandb.log(performance_log)
    else:
        keys = np.unique(np.array(s_pred), return_counts=True)[0]
        values = np.unique(np.array(s_pred), return_counts=True)[1]/s_pred.shape[0]
        s_pred_log = {'pred_'+str(i):0 for i in range(5)}
        for i in range(len(keys)):
            s_pred_log['pred_'+str(keys[i])] = values[i]
        print('----------------------------------------------------------------')
        print(f'yAcc: {y_acc:.4f} | sAcc: {s_acc:.4f}')
        print('----------------------------------------------------------------')
        # performance_log = {f'y_acc': y_acc, f's_acc': s_acc}
        # wandb.log(s_pred_log)
        # wandb.log(performance_log)
    return y_acc, s_acc

# -----------------------
# end-to-end [train & evaluation]
# -----------------------
def e2e(args, device):
    seed_everything(args.seed)
    start_time = time.time()

    y_accs = []; s_accs = []
    for it_seed in range(730, 730+args.n_seed):
        if args.n_seed != 1:
            args.seed = it_seed

        if args.data_name == 'yaleb':
            model, clf_xy, data = main(args, device)
        else:
            model, clf_xy, tr_set, te_set = main(args, device)

        eval_model = args.eval_model if args.eval_model != 'disc' else args.disc
        if args.data_name == 'yaleb':
            y_acc, s_acc = evaluation(args, device, eval_model, model, clf_xy, is_e2e=True, trainset=data, testset=data)
        else:
            y_acc, s_acc = evaluation(args, device, eval_model, model, clf_xy, is_e2e=True, trainset=tr_set, testset=te_set)
        
        y_accs.append(y_acc); s_accs.append(s_acc)
    if args.n_seed != 1:
        y_mean, y_std = np.mean(y_accs), np.std(y_accs)
        s_mean, s_std = np.mean(s_accs), np.std(s_accs)
        print('\n-----------------------------------------------------------')
        print(f"{args.n_seed} runs y acc mean: {y_mean}, std: {y_std},\n       s acc means: {s_mean}, std: {s_std}")
        print('---------------------------------------------------------')
    print('\n-----------------------------------------------------------')
    print(f"Time for Train & Eval per 1 program:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('---------------------------------------------------------')


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()
    # ------------------- run config
    parser.add_argument("--seed", type=int, default=730, help="one manual random seed")
    parser.add_argument("--n_seed", type=int, default=1, help="number of runs")
    parser.add_argument("--run_mode", type=str, default='e2e', choices=["train", "eval", "e2e"])
    parser.add_argument("--vis", type=int, default=0, help="learned representation TSNE flag")
    parser.add_argument("--vis_path", type=str, default='./TSNE/')

    # ------------------- flag & name
    parser.add_argument("--data_name", type=str, default='yaleb', help="Dataset name")
    parser.add_argument("--model_name", type=str, default='ours', help="Model ID")
    parser.add_argument("--save_name", type=str, default='default', help="specific string or system time at run start(by default)")
    parser.add_argument("--env_flag", type=str, default='nn', help="env noise flag [s,y]")
    parser.add_argument("--env_eps", type=float, default=0.15, help="env noise proportion")
    parser.add_argument("--tr_ratio", type=float, default=1.0, help="train set use rate")

    # ------------------- train config
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--scheduler", type=str, default='None', choices=['None', 'one', 'lr'], help="learning rate scheduler")
    parser.add_argument("--max_lr", type=float, default=3e-2, help="anneal max lr")
    parser.add_argument("--end_fac", type=float, default=0.001, help="linear decay end factor")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip_val", type=float, default=2.0, help="gradient clipping value for VAE backbone when using DCD loss")
    parser.add_argument("--early_stop", type=int, default=0, choices=[1, 0])
    parser.add_argument("--last_epmod", type=int, default=0, choices=[1, 0], help='use last epoch model or best model at the end of 1 stage train')
    parser.add_argument("--last_epmod_eval", type=int, default=1, choices=[1, 0], help='use last epoch model or best model for evaluation')
    parser.add_argument("--eval_model", type=str, default='lr', choices=['mlp', 'lr', 'disc'], help='representation quality evaluation model')

    # ------------------- loss config
    parser.add_argument("--kernel", type=str, default='g', choices=['t', 'g'], help="DCD loss kernel type")
    parser.add_argument("--alpha", type=float, default=1.0, help="DCD loss weight")
    parser.add_argument("--beta", type=float, default=0.2, help="KLD prior regularization weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="swap recon loss weight")

    args = parser.parse_args()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------------- specific argument setup for each dataset
    args.drop_p = 0.3
    args.neg_slop = 0.1
    args.clf_layers = 2
    args.cont_xs = 1
    if args.env_flag != 'nn':
        args.env_eps_s_tr = args.env_eps
        args.env_eps_y_tr = args.env_eps

    if args.data_name == "yaleb":
        args.y_dim = 38
        args.s_dim = 5
        args.n_features = 504
        args.latent_dim = 100
        args.hidden_units = 100
        args.encoder = 'lr'
        args.batch_size = 190
        args.batch_size_te = 1096
        args.epochs = 2000
        args.fade_in = 1  # annealing flag for DCD, SR loss
        args.beta_anneal = 1  # annealing flag for KLD loss
        args.clf_act = 'prelu'
        args.clf_seq = 'fad'
        args.clf_hidden_units = 75
        args.connection = 2
        args.enc_act = 'prelu' if args.kernel == 't' else 'gelu'
        args.dec_act = 'prelu' if args.kernel == 't' else 'gelu'
        args.enc_seq = 'fba' if args.kernel == 't' else 'fa'
        args.dec_seq = 'f'
        args.pred_act = 'leaky'
        args.pred_seq = 'fba'
        args.model_path = './model_yaleb'
        args.data_path = './data/yaleb/'
        args.result_path = './result_yaleb'
        args.clf_path = './bestclf/bestclf_yaleb.pth'
    elif args.data_name == 'adult':
        args.y_dim = 1
        args.s_dim = 1
        args.n_features = 95
        args.latent_dim = 15
        args.hidden_units = 64
        args.sensitive = 'gender_ Male'
        args.target = 'income_ >50K'
        args.train_file_name = 'adult_train_bin.csv'
        args.test_file_name = 'adult_test_bin.csv'
        args.encoder = 'mlp'
        args.batch_size = 30162
        args.batch_size_te = 15060
        args.epochs = 300
        args.fade_in = 1
        args.beta_anneal = 0
        args.clf_act = 'leaky'
        args.clf_seq = 'fad'
        args.clf_hidden_units = 64
        args.connection = 0
        args.enc_act = 'gelu'
        args.dec_act = 'gelu'
        args.enc_seq = 'fa'
        args.dec_seq = 'fa'
        args.pred_act = 'leaky'
        args.pred_seq = 'fba' if args.kernel == 't' else 'fad'
        args.model_path = './model_adult'
        args.data_path = './data/adult/'
        args.result_path = './result_adult'
        args.clf_path = './bestclf/bestclf_adult.pth'
    elif args.data_name == 'german':
        args.y_dim = 1
        args.s_dim = 1
        args.n_features = 45
        args.latent_dim = 5
        args.hidden_units = 64
        args.sensitive = 'gender_ Male'
        args.target = 'risk_Bad'
        args.train_file_name = 'german_train_bin.csv'
        args.test_file_name = 'german_test_bin.csv'
        args.encoder = 'mlp'
        args.batch_size = 800
        args.batch_size_te = 200
        args.epochs = 2000
        args.fade_in = 0
        args.beta_anneal = 1
        args.clf_act = 'leaky'
        args.clf_seq = 'fbad'
        args.clf_hidden_units = 64
        args.connection = 0
        args.enc_act = 'prelu'
        args.dec_act = 'prelu'
        args.enc_seq = 'fa'
        args.dec_seq = 'fa'
        args.pred_act = 'leaky'
        args.pred_seq = 'fba'
        args.model_path = './model_german'
        args.data_path = './data/german/'
        args.result_path = './result_german'
        args.clf_path = './bestclf/bestclf_german.pth'
    args.patience = int(args.epochs * 0.10)
    # -------------------------------------------------------------------
    # import wandb
    # wandb.init(project='FarconVAE')
    # wandb.config.update(args)

    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.isdir(args.result_path):
        os.mkdir(args.result_path)
    if not os.path.isdir(args.vis_path):
        os.mkdir(args.vis_path)

    if args.run_mode == 'train':
        main(args, device)
    elif args.run_mode == 'eval':
        evaluation(args, device, eval_model='lr', is_e2e=False)
    else:
        e2e(args, device)