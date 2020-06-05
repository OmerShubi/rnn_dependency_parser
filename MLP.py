import torch
import torch.nn as nn

from itertools import product
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(self, lstm_out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        # lstm_out_dim*2 – size of each input sample
        # self.hidden_dim - size of each output sample
        self.layer_1_h = torch.nn.Linear(lstm_out_dim*2, self.hidden_dim)
        self.layer_1_m = torch.nn.Linear(lstm_out_dim*2, self.hidden_dim)
        self.layer_2 = torch.nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Tanh()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, vectors):
        # TODO why layer h and layer m?
        seq_len = vectors.size(1)
        MLP_rep_h = (torch.zeros((seq_len, self.hidden_dim))).to(self.device)
        MLP_rep_m = (torch.zeros((seq_len, self.hidden_dim))).to(self.device)
        vectors = vectors.squeeze(0)
        for i, vec in enumerate(vectors):
            MLP_rep_h[i] = self.layer_1_h(vec)
            MLP_rep_m[i] = self.layer_1_m(vec)
        # MLP_rep_h = self.layer_1_h(vectors)
        # MLP_rep_m = self.layer_1_m(vectors)
        scores = (torch.zeros((seq_len, seq_len))).to(self.device)
        for h in range(seq_len):
            for m in range(seq_len):
                curr_addition = torch.add(MLP_rep_h[h], MLP_rep_m[m])
                scores[h, m] = self.layer_2(self.activation(curr_addition))
        return scores
