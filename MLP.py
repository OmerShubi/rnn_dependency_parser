from torch import device, cuda, add
import torch.nn as nn
from torch import manual_seed
manual_seed(0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, mlp_dropout=0):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        # self.hidden_dim - size of each output sample
        self.layer_1_h = nn.Linear(in_features=input_dim, out_features=self.hidden_dim)
        self.layer_1_m = nn.Linear(in_features=input_dim, out_features=self.hidden_dim)
        self.dropout = nn.Dropout(p=mlp_dropout, inplace=False)
        self.layer_2 = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.activation = eval(activation)()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

    def forward(self, vectors):
        # X1 = W_half1*vh, X2 = W_half2*vm
        MLP_rep_m = self.layer_1_m(vectors).to(self.device)
        MLP_rep_h = self.layer_1_h(vectors).to(self.device)

        # MLP Dropout
        MLP_rep_m = self.dropout(MLP_rep_m)
        MLP_rep_h = self.dropout(MLP_rep_h)

        # Hack for creating all h,m pairs to get W1x+b1 quickly
        MLP_rep_h1 = MLP_rep_h.squeeze().unsqueeze(1)  # [seq_len, 1, dim]
        MLP_rep_m2 = MLP_rep_m.repeat(MLP_rep_h.squeeze().shape[0], 1, 1)
        MLP_rep_h2 = MLP_rep_h1.repeat(1, MLP_rep_m.squeeze().shape[0], 1)

        # X1 + X2
        MLP_rep_matrix = add(MLP_rep_m2, MLP_rep_h2)

        # Activation + Layer 2
        scores = self.layer_2(self.activation(MLP_rep_matrix)).squeeze()  # W2*tahn(W1x+b1)+b2

        return scores
