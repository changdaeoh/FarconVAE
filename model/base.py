import torch
import torch.nn as nn
import torch.nn.functional as F


def z_sampling_(mean, log_var, device):
    eps = torch.randn(mean.size()[0], mean.size()[1], device=device)
    return eps * torch.exp(log_var / 2) + mean


class Encoder(nn.Module):
    '''One hidden layer encoder'''
    def __init__(self, input_dim, output_dim, hidden_dim, args, deterministic=False):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(args.drop_p)
        self.bn = nn.BatchNorm1d(hidden_dim)
        if deterministic:  # output point vector(z) instead of distribution parameters(mu, var)
            self.out = nn.Linear(hidden_dim, output_dim)
        else:  # output distribution parameters(mu, var)
            self.out = nn.Linear(hidden_dim, 2*output_dim)

        if args.enc_act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.neg_slop)
        elif args.enc_act == 'relu':
            self.act = nn.ReLU()
        elif args.enc_act == 'gelu':
            self.act = nn.GELU()
        elif args.enc_act == 'prelu':
            self.act = nn.PReLU()
        else:
            raise ValueError

        if args.enc_seq == 'fba':
            self.net = nn.Sequential(self.fc, self.bn, self.act)
        elif args.enc_seq == 'fad':
            self.net = nn.Sequential(self.fc, self.act, self.dropout)
        elif args.enc_seq == 'fbad':
            self.net = nn.Sequential(self.fc, self.bn, self.act, self.dropout)
        elif args.enc_seq == 'fa':
            self.net = nn.Sequential(self.fc, self.act)
        else:
            raise ValueError

        self.args = args
        self.output_dim = output_dim
        self.deterministic = deterministic

    def forward(self, inp):
        enc = self.net(inp)
        out = self.out(enc)
        if self.deterministic:
            return out    
        # Mu(x,s,y), log_Var(x,s,y)
        return out[:, :self.output_dim], out[:, self.output_dim:]


class Encoder_One(nn.Module):
    '''One layer encoder'''
    def __init__(self, input_dim, output_dim, args, deterministic=False):
        super(Encoder_One, self).__init__()
        if deterministic:  # output point vector(z) instead of distribution parameters(mu, var)
            self.fc = nn.Linear(input_dim, output_dim)
            self.bn = nn.BatchNorm1d(output_dim)
        else:  # output distribution parameters(mu, var)
            self.fc = nn.Linear(input_dim, 2 * output_dim)
            self.bn = nn.BatchNorm1d(2 * output_dim)
        self.dropout = nn.Dropout(args.drop_p)

        if args.enc_act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.neg_slop)
        elif args.enc_act == 'relu':
            self.act = nn.ReLU()
        elif args.enc_act == 'gelu':
            self.act = nn.GELU()
        elif args.enc_act == 'prelu':
            self.act = nn.PReLU()
        else:
            raise ValueError

        if args.enc_seq == 'fba':
            self.net = nn.Sequential(self.fc, self.bn, self.act)
        elif args.enc_seq == 'fad':
            self.net = nn.Sequential(self.fc, self.act, self.dropout)
        elif args.enc_seq == 'fbad':
            self.net = nn.Sequential(self.fc, self.bn, self.act, self.dropout)
        elif args.enc_seq == 'fa':
            self.net = nn.Sequential(self.fc, self.act)
        elif args.enc_seq == 'f':
            self.net = nn.Sequential(self.fc)
        else:
            raise ValueError

        self.args = args
        self.output_dim = output_dim
        self.deterministic = deterministic

    def forward(self, inp):
        out = self.net(inp)
        if self.deterministic:
            return out    
        # Mu(x,s,y), log_Var(x,s,y)
        return out[:, :self.output_dim], out[:, self.output_dim:]


class Decoder(nn.Module):
    '''VAE decoder. One hidden layer'''
    def __init__(self, input_dim, output_dim, hidden_dim, args):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(args.drop_p)
        self.bn = nn.BatchNorm1d(hidden_dim)
        if args.dec_act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.neg_slop)
        elif args.dec_act == 'relu':
            self.act = nn.ReLU()
        elif args.dec_act == 'gelu':
            self.act = nn.GELU()
        elif args.dec_act == 'prelu':
            self.act = nn.PReLU()
        else:
            raise ValueError

        if args.dec_seq == 'fba':
            self.net = nn.Sequential(self.fc, self.bn, self.act)
        elif args.dec_seq == 'fad':
            self.net = nn.Sequential(self.fc, self.act, self.dropout)
        elif args.dec_seq == 'fbad':
            self.net = nn.Sequential(self.fc, self.bn, self.act, self.dropout)
        elif args.dec_seq == 'fa':
            self.net = nn.Sequential(self.fc, self.act)
        else:
            raise ValueError

    def forward(self, latent):
        dec = self.net(latent)
        return torch.sigmoid(self.out(dec))


class Decoder_One(nn.Module):
    '''VAE decoder. One hidden layer'''
    def __init__(self, input_dim, output_dim, args):
        super(Decoder_One, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(args.drop_p)
        self.bn = nn.BatchNorm1d(input_dim)

        if args.dec_act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.neg_slop)
        elif args.dec_act == 'relu':
            self.act = nn.ReLU()
        elif args.dec_act == 'gelu':
            self.act = nn.GELU()
        elif args.dec_act == 'prelu':
            self.act = nn.PReLU()
        else:
            raise ValueError

        if args.dec_seq == 'f':
            self.net = nn.Sequential(self.fc)
        elif args.dec_seq == 'bf':
            self.net = nn.Sequential(self.bn, self.fc)
        elif args.dec_seq == 'baf':
            self.net = nn.Sequential(self.bn, self.act, self.fc)
        elif args.dec_seq == 'af':
            self.net = nn.Sequential(self.act, self.fc)
        else:
            raise ValueError

    def forward(self, latent):
        return torch.sigmoid(self.net(latent))


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, args):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(args.drop_p)
        self.bn = nn.BatchNorm1d(hidden_dim)
        if args.pred_act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.neg_slop)
        elif args.pred_act == 'relu':
            self.act = nn.ReLU()
        elif args.pred_act == 'gelu':
            self.act = nn.GELU()
        elif args.pred_act == 'prelu':
            self.act = nn.PReLU()
        else:
            raise ValueError

        if args.pred_seq == 'fbad':
            self.net = nn.Sequential(
                self.fc1, self.bn, self.act, self.dropout,
                self.fc2, self.bn, self.act, self.dropout
            )
        elif args.pred_seq == 'bfad':
            if args.data_name == 'yaleb':
                self.net = nn.Sequential(
                    self.bn, self.fc1, self.act, self.dropout,
                    self.bn, self.fc2, self.act, self.dropout,
                )
            else:
                self.bn0 = nn.BatchNorm1d(input_dim)
                self.net = nn.Sequential(
                    self.bn0, self.fc1, self.act, self.dropout,
                    self.bn, self.fc2, self.act, self.dropout,
                )
        elif args.pred_seq == 'fba':
            self.net = nn.Sequential(
                self.fc1, self.bn, self.act,
                self.fc2, self.bn, self.act
            )
        elif args.pred_seq == 'bfba':
            if args.data_name == 'yaleb':
                self.net = nn.Sequential(
                    self.bn, self.fc1, self.bn, self.act,
                    self.bn, self.fc2, self.bn, self.act
                )
            else:
                self.bn0 = nn.BatchNorm1d(input_dim)
                self.net = nn.Sequential(
                    self.bn0, self.fc1, self.bn, self.act,
                    self.bn, self.fc2, self.bn, self.act
                )
        elif args.pred_seq == 'fad':
            self.net = nn.Sequential(
                self.fc1, self.act, self.dropout,
                self.fc2, self.act, self.dropout
            )
        elif args.pred_seq == 'fa':
            self.net = nn.Sequential(
                self.fc1, self.act,
                self.fc2, self.act
            )
        else:
            raise ValueError

    def forward(self, latent):
        latent = self.net(latent)
        return self.out(latent)


class BestClf(nn.Module):
    '''
        - learn (x -> y) mapping (unfair Best classifier)
        - will be used test-time y label iput to encoder
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, args):
        super(BestClf, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.sep_conn = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(args.drop_p)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.args = args
        if args.clf_act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.neg_slop)
        elif args.clf_act == 'relu':
            self.act = nn.ReLU()
        elif args.clf_act == 'gelu':
            self.act = nn.GELU()
        elif args.clf_act == 'prelu':
            self.act = nn.PReLU()
        else:
            raise ValueError

        if args.clf_layers == 2:
            if args.clf_seq == 'fbad':
                self.net = nn.Sequential(
                        self.fc1, self.bn, self.act, self.dropout,
                        self.fc2, self.bn, self.act, self.dropout
                )
            elif args.clf_seq == 'fba':
                self.net = nn.Sequential(
                        self.fc1, self.bn, self.act,
                        self.fc2, self.bn, self.act
                )
            elif args.clf_seq == 'fad':
                self.net = nn.Sequential(
                        self.fc1, self.act, self.dropout,
                        self.fc2, self.act, self.dropout
                )
            else:
                raise ValueError
        elif args.clf_layers == 3:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            if args.clf_seq == 'fbad':
                self.net = nn.Sequential(
                        self.fc1, self.bn, self.act, self.dropout,
                        self.fc2, self.bn, self.act, self.dropout,
                        self.fc3, self.bn, self.act, self.dropout,
                )
            elif args.clf_seq == 'fba':
                self.net = nn.Sequential(
                        self.fc1, self.bn, self.act,
                        self.fc2, self.bn, self.act,
                        self.fc3, self.bn, self.act,
                )
            elif args.clf_seq == 'fad':
                self.net = nn.Sequential(
                        self.fc1, self.act, self.dropout,
                        self.fc2, self.act, self.dropout,
                        self.fc3, self.act, self.dropout,
                )
            else:
                raise ValueError
        elif args.clf_layers == 4:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, hidden_dim)
            if args.clf_seq == 'fbad':
                self.net = nn.Sequential(
                        self.fc1, self.bn, self.act, self.dropout,
                        self.fc2, self.bn, self.act, self.dropout,
                        self.fc3, self.bn, self.act, self.dropout,
                        self.fc4, self.bn, self.act, self.dropout,
                )
            elif args.clf_seq == 'fba':
                self.net = nn.Sequential(
                        self.fc1, self.bn, self.act,
                        self.fc2, self.bn, self.act,
                        self.fc3, self.bn, self.act,
                        self.fc4, self.bn, self.act,
                )
            elif args.clf_seq == 'fad':
                self.net = nn.Sequential(
                        self.fc1, self.act, self.dropout,
                        self.fc2, self.act, self.dropout,
                        self.fc3, self.act, self.dropout,
                        self.fc4, self.act, self.dropout,
                )
            else:
                raise ValueError

    # add skip connection
    def forward(self, x):
        if self.args.connection == 0:
            h = self.net(x)
        elif self.args.connection == 1:
            h = self.net(x) + self.fc1(x)
        elif self.args.connection == 2:
            h = self.net(x) + self.sep_conn(x)
        else:
            raise ValueError
        return self.out(h)


class Encoder_ODFR(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, args):
        super(Encoder_ODFR, self).__init__()
        self.encoder_shared = nn.Linear(input_dim, hidden_dim)
        self.head_t = nn.Linear(hidden_dim, output_dim * 2)
        self.head_s = nn.Linear(hidden_dim, output_dim * 2)
        self.args = args
        self.dropout = nn.Dropout(args.drop_p)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.output_dim = output_dim

        if args.enc_act == 'relu':
            self.act = nn.ReLU()
        elif args.enc_act == 'leaky':
            self.act = nn.LeakyReLU(args.neg_slop)
        elif args.enc_act == 'gelu':
            self.act = nn.GELU()
        elif args.enc_act == 'prelu':
            self.act = nn.PReLU()

        if args.enc_seq == 'fba':
            self.net = nn.Sequential(self.encoder_shared, self.bn, self.act)
        elif args.enc_seq == 'fad':
            self.net = nn.Sequential(self.encoder_shared, self.act, self.dropout)
        elif args.enc_seq == 'fbad':
            self.net = nn.Sequential(self.encoder_shared, self.bn, self.act, self.dropout)
        elif args.enc_seq == 'fa':
            self.net = nn.Sequential(self.encoder_shared, self.act)
        else:
            raise ValueError

        if args.enc_layers == 2:
            self.head_t0 = nn.Linear(hidden_dim, hidden_dim)
            self.head_s0 = nn.Linear(hidden_dim, hidden_dim)
        else:
            pass

    def forward(self, x):
        shared_enc = self.net(x)
        if self.args.enc_layers == 1:
            out_t = self.head_t(shared_enc)
            out_s = self.head_s(shared_enc)
        else:
            if self.args.enc_seq == 'fa':
                out_t = self.head_t(self.act(self.head_t0(shared_enc)))
                out_s = self.head_s(self.act(self.head_s0(shared_enc)))
            elif self.args.enc_seq == 'fad':
                out_t = self.head_t(self.dropout(self.act(self.head_t0(shared_enc))))
                out_s = self.head_s(self.dropout(self.act(self.head_s0(shared_enc))))
            elif self.args.enc_seq == 'fbad':
                out_t = self.head_t(self.dropout(self.act(self.bn(self.head_t0(shared_enc)))))
                out_s = self.head_s(self.dropout(self.act(self.bn(self.head_s0(shared_enc)))))
            elif self.args.enc_seq == 'fba':
                out_t = self.head_t(self.act(self.bn(self.head_t0(shared_enc))))
                out_s = self.head_s(self.act(self.bn(self.head_s0(shared_enc))))
        t_mu, t_lvar = out_t[:, :self.output_dim], out_t[:, self.output_dim:]
        s_mu, s_lvar = out_s[:, :self.output_dim], out_s[:, self.output_dim:]
        return t_mu, s_mu, t_lvar, s_lvar


class Encoder_ODFR_One(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(Encoder_ODFR_One, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.args = args
        self.dropout = nn.Dropout(args.drop_p)
        self.bn = nn.BatchNorm1d(output_dim * 2)
        self.output_dim = output_dim

        if args.enc_act == 'relu':
            self.act = nn.ReLU()
        elif args.enc_act == 'leaky':
            self.act = nn.LeakyReLU(args.neg_slop)
        elif args.enc_act == 'gelu':
            self.act = nn.GELU()
        elif args.enc_act == 'prelu':
            self.act = nn.PReLU()

        if args.enc_seq == 'fba':
            self.net = nn.Sequential(self.fc, self.bn, self.act)
        elif args.enc_seq == 'fad':
            self.net = nn.Sequential(self.fc, self.act, self.dropout)
        elif args.enc_seq == 'fbad':
            self.net = nn.Sequential(self.fc, self.bn, self.act, self.dropout)
        elif args.enc_seq == 'fa':
            self.net = nn.Sequential(self.fc, self.act)
        elif args.enc_seq == 'f':
            self.net = nn.Sequential(self.fc)
        else:
            raise ValueError

    def forward(self, x):
        out = self.net(x)

        t_mu, t_lvar = out[:, :self.output_dim], out[:, self.output_dim:]
        s_mu, s_lvar = out[:, :self.output_dim], out[:, self.output_dim:]
        return t_mu, s_mu, t_lvar, s_lvar