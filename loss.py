import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence


def dcd_t(d_zx, d_zs, d_xs, cont_xs=1):
    '''t kernel Distributional Contrastive Disentangle loss'''
    if cont_xs:
        dxs1, dxs2 = d_xs
        return d_zx + 1 / (1 + d_zs) + 1 / (1 + dxs1) + 1 / (1 + dxs2)
    else:
        return d_zx + 1 / (1 + d_zs)


def dcd_g(d_zx, d_zs, d_xs, cont_xs=1):
    '''G kernel Distributional Contrastive Disentangle loss'''
    if cont_xs:
        dxs1, dxs2 = d_xs
        return d_zx + torch.exp(-d_zs) + torch.exp(-dxs1) + torch.exp(-dxs2)
    else:
        return d_zx + torch.exp(-d_zs)


def kld_loss(mean, log_var, agg='mean'):
    '''VAE KLD loss'''
    if agg == 'mean':
        return -0.5 * torch.mean(1 + log_var - log_var.exp() - mean.pow(2))
    else:
        return -0.5 * (1 + log_var - log_var.exp() - mean.pow(2)).sum(1).mean()


def orth_kld_loss(mean, log_var, flag, agg='mean'):
    '''orthogonal prior loss with analytic KLD between two Gaussian'''
    one_mask = torch.ones_like(mean)
    zero_mask = torch.zeros_like(mean)
    if flag == '10':
        mask = torch.cat((one_mask[:, :(mean.shape[1]//2)], zero_mask[:, (mean.shape[1]//2):]), dim=1)
    elif flag == '01':
        mask = torch.cat((zero_mask[:, :(mean.shape[1]//2)], one_mask[:, (mean.shape[1]//2):]), dim=1)
    elif flag == '11':
        mask = one_mask
    elif flag == '1m':
        mask = torch.cat((one_mask[:, :(mean.shape[1]//2)], -one_mask[:, (mean.shape[1]//2):]), dim=1)
    elif flag == '1010':
        mask = zero_mask
        mask[:, ::2] = 1
    elif flag == '0101':
        mask = one_mask
        mask[:, ::2] = 0
    elif flag == '1111':
        mask = one_mask
    elif flag == 'm1m1':
        mask = one_mask
        mask[:, ::2] = -1
    else:
        raise ValueError
    mean_distance = mean - mask
    if agg == 'mean':
        return -0.5 * (1 + log_var - log_var.exp() - mean_distance.pow(2)).mean()
    else:
        return -0.5 * (1 + log_var - log_var.exp() - mean_distance.pow(2)).sum(dim=1).mean()


# --------------------------------
# Util functions
# --------------------------------
def kld_exact(mu_p, ss_p, mu_q, ss_q):
    '''kld computing function'''
    p = MultivariateNormal(mu_p, torch.diag_embed(ss_p))
    q = MultivariateNormal(mu_q, torch.diag_embed(ss_q))
    return (kl_divergence(p, q)/mu_p.shape[1]).mean()


def compute_bi_kld(mu_x1, logvar_x1, mu_s1, logvar_s1, mu_x2, logvar_x2, mu_s2, logvar_s2):
    '''averaged kld (symmetrize)'''
    ss_x1, ss_s1, ss_x2, ss_s2 = logvar_x1.exp(), logvar_s1.exp(), logvar_x2.exp(), logvar_s2.exp()

    # Div(Zx, Zx') - must be sim
    d_zx_l = kld_exact(mu_x1, ss_x1, mu_x2, ss_x2)
    d_zx_r = kld_exact(mu_x2, ss_x2, mu_x1, ss_x1)
    d_zx = (d_zx_l + d_zx_r) / 2

    # Div(Zs, Zs') - must be diff
    d_zs_l = kld_exact(mu_s1, ss_s1, mu_s2, ss_s2)
    d_zs_r = kld_exact(mu_s2, ss_s2, mu_s1, ss_s1)
    d_zs = (d_zs_l + d_zs_r) / 2

    if mu_x1.shape == mu_s1.shape:  # Zx <-> Zs 
        # Div(Zx, Zs) - must be diff
        d_xs_ori_l = kld_exact(mu_x1, ss_x1, mu_s1, ss_s1)
        d_xs_ori_r = kld_exact(mu_s1, ss_s1, mu_x1, ss_x1)
        d_xs_ori = (d_xs_ori_l + d_xs_ori_r) / 2

        # Div(Zx', Zs') - must be diff
        d_xs_cont_l = kld_exact(mu_x2, ss_x2, mu_s2, ss_s2)
        d_xs_cont_r = kld_exact(mu_s2, ss_s2, mu_x2, ss_x2)
        d_xs_cont = (d_xs_cont_l + d_xs_cont_r) / 2

        d_xs = d_xs_ori, d_xs_cont
        # tensor, tensor, tensor tuple
        return d_zx, d_zs, d_xs
    else:
        return d_zx, d_zs, torch.tensor(0.0)

class FarconVAELoss(nn.Module):
    def __init__(self, args, device, total_train_it):
        super(FarconVAELoss, self).__init__()
        # config
        self.args = args
        self.device = device
        self.total_train_it = total_train_it
        
        # loss
        self.kld_loss = kld_loss
        self.recon_lossf = nn.BCELoss()  # recon(z, dec(enc(z)))
        self.pred_lossf = nn.BCEWithLogitsLoss() if args.data_name != 'yaleb' else nn.CrossEntropyLoss()  # y prediction loss 

    def forward(self, out1, out2, x_ori, x_cont, s_ori, s_cont, y, model, current_iter, is_train=True):
        if is_train:
            # 1.1. ELBO - Reconstruction Loss
            recon_x_loss1, recon_x_loss2 = self.recon_lossf(out1[1][0], x_ori), self.recon_lossf(out2[1][0], x_cont)
            recon_s_loss1, recon_s_loss2 = self.recon_lossf(out1[1][1], s_ori.float()), self.recon_lossf(out2[1][1], s_cont.float())
            recon_x_loss = 0.5 * (recon_x_loss1 + recon_x_loss2)
            recon_s_loss = 0.5 * (recon_s_loss1 + recon_s_loss2)
            recon_loss = recon_x_loss + recon_s_loss
            
            # 1.2. ELBO - KL Regularization Loss
            kl_loss_x1, kl_loss_s1 = self.kld_loss(out1[2][0], out1[2][1]), self.kld_loss(out1[2][2], out1[2][3])
            kl_loss_x2, kl_loss_s2 = self.kld_loss(out2[2][0], out2[2][1]), self.kld_loss(out2[2][2], out2[2][3])
            kl_loss_1 = 0.5 * (kl_loss_x1 + kl_loss_s1)
            kl_loss_2 = 0.5 * (kl_loss_x2 + kl_loss_s2)
            kl_loss = 0.5 * (kl_loss_1 + kl_loss_2)

            # 1.3. ELBO - Prediction loss
            y_hat_1, y_hat_2 = out1[3], out2[3]
            y_hat = y_hat_1.detach().clone()
            pred_loss = (self.pred_lossf(y_hat_1, y) + self.pred_lossf(y_hat_2, y))/2
            if self.args.data_name != 'yaleb':
                y_pred = torch.ones_like(y_hat, device=self.device)
                y_pred[y_hat < 0] = 0.0
            else:
                y_pred = torch.argmax(y_hat, dim=1)

            # 2. Contrastive Loss
            d_zx, d_zs, d_xs, cont_loss = 0.0, 0.0, 0.0, 0.0
            if self.args.alpha != 0:
                d_zx, d_zs, d_xs = compute_bi_kld(out1[2][0], out1[2][1], out1[2][2], out1[2][3], out2[2][0], out2[2][1], out2[2][2], out2[2][3])
                if self.args.kernel == 't':
                    cont_loss = dcd_t(d_zx, d_zs, d_xs, self.args.cont_xs)
                elif self.args.kernel == 'g':
                    cont_loss = dcd_g(d_zx, d_zs, d_xs, self.args.cont_xs)

            # 3. Swap Reconstruction Loss
            sr_loss = 0
            if self.args.gamma != 0:
                x_mix_rec1, s_mix_rec1 = model.decode(out2[0][0], out1[0][1])
                x_mix_rec2, s_mix_rec2 = model.decode(out1[0][0], out2[0][1])
                ms_ori = self.recon_lossf(torch.cat((x_mix_rec1, s_mix_rec1), dim=1), torch.cat((x_ori, s_ori.float()), dim=1))
                ms_cont = self.recon_lossf(torch.cat((x_mix_rec2, s_mix_rec2), dim=1), torch.cat((x_cont, s_cont.float()), dim=1))
                sr_loss = (ms_ori + ms_cont) / 2
            
            # Annealing Strategy
            temperature = 1.0
            if self.args.fade_in:
                if current_iter < (self.total_train_it//10):
                    temperature = 1/(1 + np.exp(-(current_iter - (self.total_train_it//20))))
            
            beta_temperature = 1.0
            if self.args.beta_anneal:
                if current_iter < (self.total_train_it//10):
                    beta_temperature = 1/(1 + np.exp(-(current_iter - (self.total_train_it//20))))

            # final loss
            kl_loss = kl_loss * beta_temperature * self.args.beta
            cont_loss = cont_loss * temperature * self.args.alpha
            sr_loss = sr_loss * temperature * self.args.gamma
            return recon_loss, kl_loss, pred_loss, cont_loss, sr_loss, y_pred
        else:
            # recon loss
            recon_x_loss = self.recon_lossf(out1[1][0], x_ori)
            recon_s_loss = self.recon_lossf(out1[1][1], s_ori.float())

            # contrastive loss
            d_zx, d_zs, d_xs, cont_loss = 0.0, 0.0, 0.0, 0.0
            if self.args.alpha != 0:
                d_zx, d_zs, d_xs = compute_bi_kld(out1[2][0], out1[2][1], out1[2][2], out1[2][3], out2[2][0], out2[2][1], out2[2][2], out2[2][3])
                if self.args.kernel == 't':
                    cont_loss = dcd_t(d_zx, d_zs, d_xs, self.args.cont_xs)
                elif self.args.kernel == 'g':
                    cont_loss = dcd_g(d_zx, d_zs, d_xs, self.args.cont_xs)

            # prediction loss
            pred_loss = self.pred_lossf(out1[3], y)
            
            if self.args.data_name != 'yaleb':
                y_pred = torch.ones_like(out1[3], device=self.device)
                y_pred[out1[3] < 0] = 0.0
            else:
                y_pred = torch.argmax(out1[3], dim=1)
            return recon_x_loss, recon_s_loss, pred_loss, cont_loss, y_pred


# --------------------------------
# ODFR / Entropy Loss
# --------------------------------
def entropy_loss(logit, agg, data_name):
    if data_name != 'yaleb':
        p = torch.sigmoid(logit)
        ent = p * torch.log(p + 1e-9) + (1-p) * torch.log(1-p + 1e-9)
        return (ent).mean() if agg == 'sum' else (ent/2).mean()
    else:
        ent = torch.softmax(logit, dim=1) * torch.log_softmax(logit, dim=1)
        return (ent).sum(dim=1).mean() if agg == 'sum' else ent.mean()

class Criterion(nn.Module):
    def __init__(self, args, device):
        super(Criterion, self).__init__()
        self.args = args
        self.device = device

        self.lambda_e = args.lambda_e
        self.lambda_od = args.lambda_od
        self.gamma_e = args.gamma_e
        self.gamma_od = args.gamma_od
        self.step_size = args.step_size

        if args.data_name == 'yaleb':
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, target, current_step):
        t_mu, s_mu, t_lvar, s_lvar = inputs[1]
        zs_s, zt_y, zt_s = inputs[2]

        # target prediction loss
        L_t = self.ce(zt_y, target)

        # entropy loss
        Loss_e = entropy_loss(zt_s, self.args.kld_agg, self.args.data_name)

        # prior loss (OD or Not)
        if self.args.orthogonal_prior:
            if self.args.orth_ver == 1:
                L_zt = orth_kld_loss(mean=t_mu, log_var=t_lvar, flag='10', agg=self.args.kld_agg)
                L_zs = orth_kld_loss(mean=s_mu, log_var=s_lvar, flag='01', agg=self.args.kld_agg)
            elif self.args.orth_ver == 2:
                L_zt = orth_kld_loss(mean=t_mu, log_var=t_lvar, flag='11', agg=self.args.kld_agg)
                L_zs = orth_kld_loss(mean=s_mu, log_var=s_lvar, flag='1m', agg=self.args.kld_agg)
            elif self.args.orth_ver == 3:
                L_zt = orth_kld_loss(mean=t_mu, log_var=t_lvar, flag='1010', agg=self.args.kld_agg)
                L_zs = orth_kld_loss(mean=s_mu, log_var=s_lvar, flag='0101', agg=self.args.kld_agg)
            elif self.args.orth_ver == 4:
                L_zt = orth_kld_loss(mean=t_mu, log_var=t_lvar, flag='1111', agg=self.args.kld_agg)
                L_zs = orth_kld_loss(mean=s_mu, log_var=s_lvar, flag='m1m1', agg=self.args.kld_agg)
            else:
                raise ValueError
        else:
            L_zt = kld_loss(mean=t_mu, log_var=t_lvar, agg=self.args.kld_agg)
            L_zs = kld_loss(mean=s_mu, log_var=s_lvar, agg=self.args.kld_agg)

        lambda_e = self.lambda_e * (self.gamma_e ** (current_step / self.step_size))
        lambda_od = self.lambda_od * (self.gamma_od ** (current_step / self.step_size))
        tot_Loss = L_t + lambda_e * Loss_e + lambda_od * (L_zt + L_zs)

        return tot_Loss, L_t, lambda_e * Loss_e, lambda_od * (L_zt + L_zs)