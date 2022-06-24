import torch
import torch.nn as nn
from model.base import Encoder_ODFR, Discriminator, Encoder_ODFR_One
from model.eval_model import OneLinearLayer

# Orthogonal Disentangled Fair Representation
class ODFR(nn.Module):
    def __init__(self, args, device):
        super(ODFR, self).__init__()
        if args.encoder == 'mlp':
            self.encoder = Encoder_ODFR(args.n_features, args.latent_dim, args.hidden_units, args)
        else:
            self.encoder = Encoder_ODFR_One(args.n_features, args.latent_dim, args)
        if args.disc_y == "mlp":
            self.disc_y = Discriminator(args.latent_dim, args.y_dim, args.hidden_units, args)
        else:
            self.disc_y = OneLinearLayer(args.latent_dim, args.y_dim)
        self.disc_s = Discriminator(args.latent_dim, args.s_dim, args.hidden_units, args)
        self.args = args
        self.device = device

    def forward(self, x):
        t_mu, s_mu, t_lvar, s_lvar = self.encoder(x)
        zt, zs = self.z_sampling_(t_mu, t_lvar, s_mu, s_lvar)
        zs_s, zt_y, zt_s = self.disc_s(zs), self.disc_y(zt), self.disc_s(zt)
        return (zt, zs), (t_mu, s_mu, t_lvar, s_lvar), (zs_s, zt_y, zt_s)

    def z_sampling_(self, mu_t, lvar_t, mu_s, lvar_s):
        eps = torch.randn(mu_t.size()[0], mu_t.size()[1], device=self.device) 
        zt = eps * torch.exp(lvar_t / 2) + mu_t
        zs = eps * torch.exp(lvar_s / 2) + mu_s
        return zt, zs