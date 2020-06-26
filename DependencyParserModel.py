from chu_liu_edmonds import decode_mst
from utils.DataPreprocessing import *
from MLP import *
from contextlib import nullcontext
import random
from torch import manual_seed
manual_seed(0)


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, word_dict, tag_dict, word_list, tag_list,
                 tag_embedding_dim, word_embedding_dim, pretrained_embedding, lstm_hidden_dim,
                 mlp_hidden_dim, bilstm_layers, dropout_alpha, activation, freeze_embedding, lstm_dropout, mlp_dropout):
        super(KiperwasserDependencyParser, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dropout = dropout_alpha
        # this is the word and tag dicts from the data that the model trained on.
        # Used for inference
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.tag_list = tag_list
        self.word_list = word_list
        self.unknown_word_idx = 1
        self.unknown_tag_idx = 1
        self.root_idx = 0
        self.dropout = dropout_alpha
        self.word_embedding_dim = word_embedding_dim

        if pretrained_embedding is not None:
            self.word_embedder = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
        else:
            self.word_embedder = nn.Embedding(len(self.word_list), int(word_embedding_dim))

        self.tag_embedder = nn.Embedding(len(self.tag_list), tag_embedding_dim)

        self.emb_dim = self.word_embedder.embedding_dim + self.tag_embedder.embedding_dim

        self.lstm_hidden_dim = lstm_hidden_dim if lstm_hidden_dim else self.emb_dim

        self.encoder = nn.LSTM(input_size=self.emb_dim,
                               hidden_size=self.lstm_hidden_dim,
                               num_layers=bilstm_layers,
                               dropout=lstm_dropout,
                               bidirectional=True,
                               batch_first=True)

        # input samples dim for MLP is lstm_out_dim=lstm_hidden_dim*NUM_DIRECTION
        self.mlp_edge_scorer = MLP(self.lstm_hidden_dim * 2, mlp_hidden_dim, activation, mlp_dropout=mlp_dropout)

        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, sentence, is_test=False, is_comp=False):
        cm = torch.no_grad() if (is_test or is_comp) else nullcontext()
        with cm:
            word_idx_tensor, tag_idx_tensor, true_tree_heads = sentence

            if self.dropout and not (is_test or is_comp):
                for i, word in enumerate(word_idx_tensor[0]):
                    actual_word_idx = word.item()
                    if actual_word_idx != self.unknown_word_idx and actual_word_idx != self.root_idx:
                        freq_of_word = self.word_dict[self.word_list[actual_word_idx]]
                        prob_word = float(self.dropout) / (self.dropout + freq_of_word)
                        if random.random() < prob_word:
                            word_idx_tensor[0, i] = self.unknown_word_idx
                            tag_idx_tensor[0, i] = self.unknown_tag_idx

            # Pass word_idx and tag_idx through their embedding layers
            tag_embbedings = self.tag_embedder(tag_idx_tensor.to(self.device))
            word_embbedings = self.word_embedder(word_idx_tensor.to(self.device))

            # Concat both embedding outputs
            input_embeddings = torch.cat((word_embbedings, tag_embbedings), dim=2)

            # Get Bi-LSTM hidden representation for each word+tag in sentence
            lstm_output, _ = self.encoder(input_embeddings.view(1, input_embeddings.shape[1], -1))

            # Get score for each possible edge in the parsing graph, construct score matrix
            scores = self.mlp_edge_scorer(lstm_output)

            # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
            seq_len = lstm_output.size(1)
            predicted_tree_heads, _ = self.decoder(scores.data.cpu().numpy(), seq_len, False)

            if not is_comp:
                true_tree_heads = true_tree_heads.squeeze(0)
                # Calculate the negative log likelihood loss described above
                loss = self.criterion(torch.transpose(scores, 0, 1), true_tree_heads.to(self.device))
                return loss, torch.from_numpy(predicted_tree_heads)

            else:
                return torch.from_numpy(predicted_tree_heads)
