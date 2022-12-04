from torch import nn


class Mlp(nn.Module):
    def __init__(self, embed_dim, ext_rate=4, act_layer=nn.GELU, drop=0.0):
        super(Mlp, self).__init__()
        hidden_dim = int(embed_dim * ext_rate)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=hidden_dim)
        self.activate = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        shortcat = x
        x = self.activate(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x + shortcat)
