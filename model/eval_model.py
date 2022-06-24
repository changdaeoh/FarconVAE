import torch.nn as nn

class OneLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLinearLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)