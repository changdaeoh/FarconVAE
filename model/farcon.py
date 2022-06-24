from model.base import *
from model.eval_model import OneLinearLayer

class FarconVAE(nn.Module):
    def __init__(self, args, device):
        super(FarconVAE, self).__init__()
        input_dim = args.n_features + args.s_dim + args.y_dim
        if args.encoder == 'mlp':
            self.encoder = Encoder(input_dim, 2 * args.latent_dim, args.hidden_units, args)
            self.decoder = Decoder(2 * args.latent_dim, args.n_features + args.s_dim, args.hidden_units, args)
        else:
            self.encoder = Encoder_One(input_dim, 2 * args.latent_dim, args)
            self.decoder = Decoder_One(2 * args.latent_dim, args.n_features + args.s_dim, args)
        
        self.predictor = OneLinearLayer(args.latent_dim, args.y_dim)

        self.args = args
        self.device = device

    def encode(self, x, s, y, input_label=False, return_z=False):
        if input_label:
            y = F.one_hot(y.to(torch.int64), num_classes=38)
        model_input = torch.cat((x, s, y), dim=1)

        out1, out2 = self.encoder(model_input)
        (mu_x, mu_s, logvar_x, logvar_s) = out1[:, :self.args.latent_dim], out1[:, self.args.latent_dim:], \
                                           out2[:, :self.args.latent_dim], out2[:, self.args.latent_dim:]
        zx = z_sampling_(mu_x, logvar_x, self.device)
        zs = z_sampling_(mu_s, logvar_s, self.device)
        if return_z:
            return torch.cat((zx, zs), dim=1)
        return (zx, zs), (mu_x, logvar_x, mu_s, logvar_s)

    def decode(self, zx, zs):
        recon = self.decoder(torch.cat((zx, zs), dim=1))
        x_recon = recon[:, :-self.args.s_dim]
        s_recon = recon[:, -self.args.s_dim:].reshape(-1, self.args.s_dim)
        return x_recon, s_recon

    def predict(self, zx):
        return self.predictor(zx)

    def forward(self, x, s, y, input_label=False):
        (zx, zs), (mu_x, logvar_x, mu_s, logvar_s) = self.encode(x, s, y, input_label)
        x_recon, s_recon = self.decode(zx, zs)
        y_pred = self.predictor(zx)
        return (zx, zs), (x_recon, s_recon), (mu_x, logvar_x, mu_s, logvar_s), y_pred

    def vae_params(self):
        """Returns VAE parameters required for training VAE"""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def pred_params(self):
        """Returns predictor parameters"""
        return list(self.predictor.parameters())

    def get_optimizer(self):
        """Returns an optimizer for each network"""
        optimizer_vae = torch.optim.Adam(self.vae_params(), lr=self.args.lr, weight_decay=self.args.wd, amsgrad=1)
        optimizer_pred = torch.optim.Adam(self.pred_params(), lr=self.args.lr, weight_decay=self.args.wd, amsgrad=1)
        return optimizer_vae, optimizer_pred