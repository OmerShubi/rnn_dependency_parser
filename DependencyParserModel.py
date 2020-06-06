from chu_liu_edmonds import decode_mst
import torch.nn as nn
from utils.DataPreprocessing import *
from MLP import *
from contextlib import nullcontext



class KiperwasserDependencyParser(nn.Module):
    # TODO lstm_out_dim use
    def __init__(self, word_vocab_size, tag_vocab_size, tag_embedding_dim=25, word_embedding_dim=100,
                 lstm_out_dim=None, word_embeddings=None, hidden_dim=None, hidden_dim_mlp=100):
        super(KiperwasserDependencyParser, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if word_embeddings:
            self.word_embedder = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        else:
            self.word_embedder = nn.Embedding(word_vocab_size, word_embedding_dim)

        self.tag_embedder = nn.Embedding(tag_vocab_size, tag_embedding_dim)

        self.emb_dim = self.word_embedder.embedding_dim + self.tag_embedder.embedding_dim

        self.lstm_out_dim = lstm_out_dim if lstm_out_dim else self.emb_dim

        self.hidden_dim = hidden_dim if hidden_dim else self.emb_dim

        self.encoder = nn.LSTM(input_size=self.emb_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=2,
                               bidirectional=True,
                               batch_first=True)

        # input samples dim for MLP is lstm_out_dim*NUM_DIRECTION
        self.edge_scorer = MLP(self.lstm_out_dim*2, hidden_dim_mlp)

        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference

        self.log_soft_max = nn.LogSoftmax(dim=0)

    def forward(self, sentence):
        loss, predicted_tree = self.infer(sentence)

        return loss, predicted_tree

    def infer(self, sentence, is_comp=False):
        cm = torch.no_grad() if is_comp else nullcontext()
        with cm:
            word_idx_tensor, tag_idx_tensor, true_tree_heads = sentence

            # Pass word_idx and tag_idx through their embedding layers
            tag_embbedings = self.tag_embedder(tag_idx_tensor.to(self.device))
            word_embbedings = self.word_embedder(word_idx_tensor.to(self.device))

            # Concat both embedding outputs
            input_embeddings = torch.cat((word_embbedings, tag_embbedings), dim=2)

            # Get Bi-LSTM hidden representation for each word+tag in sentence
            lstm_output, _ = self.encoder(input_embeddings.view(1, input_embeddings.shape[1], -1))

            # Get score for each possible edge in the parsing graph, construct score matrix
            scores = self.edge_scorer(lstm_output)

            # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
            seq_len = lstm_output.size(1)
            predicted_tree, _ = self.decoder(scores.data.cpu().numpy(), seq_len, False)

            if not is_comp:
                true_tree_heads = true_tree_heads.squeeze(0)
                # Calculate the negative log likelihood loss described above
                probs_logged = self.log_soft_max(scores)
                loss = KiperwasserDependencyParser.nll_loss(probs_logged, true_tree_heads, self.device)
                return loss, torch.from_numpy(predicted_tree)

            else:
                return torch.from_numpy(predicted_tree)

    @staticmethod
    def nll_loss(scores, tree, device):
        loss = torch.tensor(0, dtype=torch.float).to(device)
        tree_length = tree.size(0)
        for m, h in enumerate(tree):
            loss -= scores[h, m]
        return loss / tree_length


