from model.base import *
from model.eval_model import OneLinearLayer

# MaxEnt and CI
class MaxEnt(nn.Module):
    def __init__(self, args, device):
        super(MaxEnt, self).__init__()
        if args.encoder == 'mlp':
            self.encoder = Encoder(args.n_features, args.latent_dim, args.hidden_units, args, deterministic=True)
        else:
            self.encoder = Encoder_One(args.n_features, args.latent_dim, args, deterministic=True)
        if args.disc_y == 'mlp':
            self.target_net = Predictor(args.latent_dim, args.y_dim, args.hidden_units, args)
        else:
            self.target_net = OneLinearLayer(args.latent_dim, args.y_dim)
        self.discriminator = Predictor(args.latent_dim, args.s_dim, args.hidden_units, args)

        self.args = args
        self.device = device

    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.target_net(z)
        s_pred = self.discriminator(z)
        
        return z, y_pred, s_pred