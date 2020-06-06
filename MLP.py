import torch
import torch.nn as nn

from itertools import product
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


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
        seq_len = vectors.size(1) # sentence len - num of word in sen
        MLP_rep_h = (torch.zeros((seq_len, self.hidden_dim))).to(self.device)
        MLP_rep_m = (torch.zeros((seq_len, self.hidden_dim))).to(self.device)
        vectors = vectors.squeeze(0)

        for i, vec in enumerate(vectors):
            MLP_rep_h[i] = self.layer_1_h(vec)
            MLP_rep_m[i] = self.layer_1_m(vec)

        # TODO matrix to layer -> change dim in layer
        # MLP_rep_h = self.layer_1_h(vectors)
        # MLP_rep_m = self.layer_1_m(vectors)

        scores = (torch.zeros((seq_len, seq_len))).to(self.device)
        for h in range(seq_len):
            for m in range(seq_len):
                curr_addition = torch.add(MLP_rep_h[h], MLP_rep_m[m]) # W_half1*vh +W_half2*vm
                scores[h, m] = self.layer_2(self.activation(curr_addition)) # W2*tahn(W1x+b1)+b2
        return scores
