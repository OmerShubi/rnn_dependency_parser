from torch.utils.data.dataset import Dataset
from collections import Counter, defaultdict
from torchtext.vocab import Vocab
import torch

ROOT_WORD = "<root>"
ROOT_TAG = "<root>"
UNKNOWN_TOKEN = "<unk>"
UNKNOWN_TAG = "<unk>"

SPECIAL_TOKENS = [ROOT_WORD, UNKNOWN_TOKEN]
PRETRAINED_VECTOR_EMBEDDING = "glove.6B.100d"
def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack

# TODO min and max thresholds for word occurences
def get_vocabs(list_of_paths):
    """
        TODO Fix docs
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx number of count of each tag
    """
    word_dict = {}
    pos_dict = {}
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                splited_words = line.split()

                if splited_words:
                    curr_word = splited_words[1]
                    curr_tag = splited_words[3]

                    word_dict[curr_word] = word_dict.get(curr_word, 0) + 1
                    pos_dict[curr_tag] = pos_dict.get(curr_tag, 0) + 1

    return word_dict, pos_dict


class DepDataReader:
    def __init__(self, file, word_dict, pos_dict,word_idx_mappings,pos_idx_mappings,comp=False):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.word_idx_mappings = word_idx_mappings
        self.pos_idx_mappings = pos_idx_mappings
        self.comp = comp
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        self.sentences = []
        word_dict_keys = list(self.word_idx_mappings.keys())
        pos_dict_keys = list(self.pos_idx_mappings.keys())

        with open(self.file) as f:
            # TODO fix typo
            # seen = indexes of seen words and tags
            sen_words = [self.word_idx_mappings.get(ROOT_WORD)]
            sen_poses = [self.pos_idx_mappings.get(ROOT_TAG)]
            sen_heads = [-1]
            for line in f:
                splited_words = line.split()
                # empty row --> dump the previous sentence and start again
                if not splited_words:
                    self.sentences.append((torch.tensor(sen_words, dtype=torch.long, requires_grad=False),
                                           torch.tensor(sen_poses, dtype=torch.long, requires_grad=False),
                                           torch.tensor(sen_heads, dtype=torch.long, requires_grad=False)))

                    sen_words = [self.word_idx_mappings.get(ROOT_WORD)]
                    sen_poses = [self.pos_idx_mappings.get(ROOT_TAG)]
                    sen_heads = [-1]
                # part of sentence
                else:
                    curr_word = splited_words[1]
                    curr_pos = splited_words[3]
                    if self.comp:
                        curr_head = -1
                    else:
                        curr_head = int(splited_words[6])

                    if curr_word in word_dict_keys:
                        curr_word_idx = self.word_idx_mappings.get(curr_word)
                    else:
                        curr_word_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)

                    if curr_pos in pos_dict_keys:
                        curr_pos_idx = self.pos_idx_mappings.get(curr_pos)
                    else:
                        curr_pos_idx = self.pos_idx_mappings.get(UNKNOWN_TAG)

                    sen_words.append(curr_word_idx)
                    sen_poses.append(curr_pos_idx)
                    sen_heads.append(curr_head)

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class DepDataset(Dataset):
    def __init__(self, word_dict, pos_dict, file_path: str, word_embedding_dim=-1, padding=False, comp=False):
        super().__init__()
        self.file = file_path
        self.vocab_size = len(word_dict)  # number of words in vocabulary
        self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
            word_dict, word_embedding_dim)
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(pos_dict)
        self.datareader = DepDataReader(self.file, word_dict, pos_dict, self.word_idx_mappings, self.pos_idx_mappings, comp) # todo dont keep datareader
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)  # todo always 1?
        self.unknown_pos_idx = self.pos_idx_mappings.get(UNKNOWN_TAG)  # todo always 1?
        self.root_idx = self.word_idx_mappings.get(ROOT_WORD)  # todo always 0?
        # TODO one line if
        if self.word_vectors is None:
            self.word_vector_dim = word_embedding_dim
        else:
            self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence[0]) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.datareader.sentences

    def __len__(self):
        return len(self.sentences_dataset)  # todo self.number of sentences

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, head_indexes_in_sentence = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, head_indexes_in_sentence

    @staticmethod
    def init_word_embeddings(word_dict, word_embedding_dim):
        # TODO add freq_min?
        if word_embedding_dim <= 0:
            glove = Vocab(Counter(word_dict), vectors=PRETRAINED_VECTOR_EMBEDDING, specials=SPECIAL_TOKENS)
        else:
            glove = Vocab(Counter(word_dict), vectors=None, specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors
    # todo delete - no use
    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        # TODO clean up sort
        # TODO not working with comp
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS]) # will not work with comp [0,1]

        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))

        return pos_idx_mappings, idx_pos_mappings

    # todo delete - no use
    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings







