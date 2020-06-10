import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        # self.hidden_dim - size of each output sample
        self.layer_1_h = torch.nn.Linear(in_features=input_dim, out_features=self.hidden_dim)
        self.layer_1_m = torch.nn.Linear(in_features=input_dim, out_features=self.hidden_dim)
        self.layer_2 = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.activation = nn.Tanh()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, vectors):

        # X1 = W_half1*vh, X2 = W_half2*vm
        MLP_rep_m = self.layer_1_m(vectors).to(self.device)
        MLP_rep_h = self.layer_1_h(vectors).to(self.device)

        # Hack for creating all h,m pairs to get W1x+b1 quickly
        MLP_rep_h1 = MLP_rep_h.squeeze().unsqueeze(1)  # [seq_len, 1, dim]
        MLP_rep_m2 = MLP_rep_m.repeat(MLP_rep_h.squeeze().shape[0], 1, 1)
        MLP_rep_h2 = MLP_rep_h1.repeat(1, MLP_rep_m.squeeze().shape[0], 1)

        # X1 + X2
        MLP_rep_matrix = torch.add(MLP_rep_m2, MLP_rep_h2)

        # Activation + Layer 2
        scores = self.layer_2(self.activation(MLP_rep_matrix)).squeeze()# W2*tahn(W1x+b1)+b2
        return scores
