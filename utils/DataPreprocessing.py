from torch.utils.data.dataset import Dataset
from collections import Counter, defaultdict
from torchtext.vocab import Vocab
import torch

ROOT_WORD = ROOT_TAG = "<root>"
UNKNOWN_TOKEN = UNKNOWN_TAG = "<unk>"
SPECIAL_TOKENS = [ROOT_WORD, UNKNOWN_TOKEN]
PRETRAINED_VECTOR_EMBEDDING = "glove.6B.100d"

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
    tag_dict = {}
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                splited_words = line.split()

                if splited_words:
                    curr_word = splited_words[1]
                    curr_tag = splited_words[3]

                    word_dict[curr_word] = word_dict.get(curr_word, 0) + 1
                    tag_dict[curr_tag] = tag_dict.get(curr_tag, 0) + 1

    return word_dict, tag_dict

class DepDataReader:
    # TODO comp use
    def __init__(self, file, word_dict, tag_dict ,word_to_idx_dict ,tag_to_idx_dict ,comp=False):
        self.file = file
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.sentences = []
        self.word_to_idx_dict = word_to_idx_dict
        self.tag_to_idx_dict = tag_to_idx_dict
        self.comp = comp
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        self.sentences = []
        unknown_token_idx = self.word_to_idx_dict.get(UNKNOWN_TOKEN)
        unknown_tag_idx = self.tag_to_idx_dict.get(UNKNOWN_TAG)

        with open(self.file) as f:
            # seen = indexes of seen words and tags
            seen_words = [self.word_to_idx_dict.get(ROOT_WORD)]
            seen_tags = [self.tag_to_idx_dict.get(ROOT_TAG)]
            seen_heads = [-1]

            for line in f:
                splited_words = line.split()

                # empty row --> dump the previous sentence and start again
                if not splited_words:
                    self.sentences.append((torch.tensor(seen_words, dtype=torch.long, requires_grad=False),
                                           torch.tensor(seen_tags, dtype=torch.long, requires_grad=False),
                                           torch.tensor(seen_heads, dtype=torch.long, requires_grad=False)))

                    seen_words = [self.word_to_idx_dict.get(ROOT_WORD)]
                    seen_tags = [self.tag_to_idx_dict.get(ROOT_TAG)]
                    seen_heads = [-1]

                # part of sentence
                else:
                    curr_word = splited_words[1]
                    curr_word_idx = self.word_to_idx_dict.get(curr_word, unknown_token_idx)

                    curr_tag = splited_words[3]
                    curr_tag_idx = self.tag_to_idx_dict.get(curr_tag, unknown_tag_idx)

                    # TODO comp use
                    curr_head = -1 if self.comp else int(splited_words[6])

                    seen_words.append(curr_word_idx)
                    seen_tags.append(curr_tag_idx)
                    seen_heads.append(curr_head)

    # todo delete - no use
    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class DepDataset(Dataset):
    # TODO padding use?
    # TODO comp use
    def __init__(self, word_dict, tag_dict, file_path: str, word_embedding_dim =- 1, padding=False, comp=False):
        super().__init__()
        self.file = file_path
        self.word_vocab_size = len(word_dict)  # number of words in vocabulary
        self.word_to_idx_dict, self.words_list, self.word_vectors = self.init_word_embeddings(
            word_dict, word_embedding_dim)
        self.tag_to_idx_dict, self.tags_list = self.init_tag_vocab(tag_dict)
        self.unknown_idx = self.word_to_idx_dict.get(UNKNOWN_TOKEN)
        self.unknown_tag_idx = self.tag_to_idx_dict.get(UNKNOWN_TAG)
        self.root_idx = self.word_to_idx_dict.get(ROOT_WORD)
        # word_embedding_dim == -1 -> self.word_vector_dim is determind by the PRETRAINED_VECTOR_EMBEDDING name
        self.word_vector_dim = word_embedding_dim if self.word_vectors is None else self.word_vectors.size(-1)
        datareader = DepDataReader(self.file, word_dict, tag_dict, self.word_to_idx_dict, self.tag_to_idx_dict, comp)
        self.sentences_dataset = datareader.sentences
        sentences_lens = [len(sentence[0])-1 for sentence in self.sentences_dataset]
        self.num_edges = sum(sentences_lens) # num of words
        self.max_seq_len = max(sentences_lens) + 1 # +1 for root # TODO delete if no padding
        self.num_sentences = len(self.sentences_dataset)

    def __len__(self):
        """returns num of sentences in data"""
        return self.num_sentences

    def __getitem__(self, index):
        word_embed_idx, tag_embed_idx, head_indexes_in_sentence = self.sentences_dataset[index]
        return word_embed_idx, tag_embed_idx, head_indexes_in_sentence

    @staticmethod
    def init_word_embeddings(word_dict, word_embedding_dim):
        # TODO add freq_min?
        if word_embedding_dim <= 0:
            glove = Vocab(Counter(word_dict), vectors=PRETRAINED_VECTOR_EMBEDDING, specials=SPECIAL_TOKENS)
        else:
            # advanced model
            glove = Vocab(Counter(word_dict), vectors=None, specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    @staticmethod
    def init_tag_vocab(tag_dict):
        # TODO add freq_min?
        # TODO pretrainded?
        glove = Vocab(Counter(tag_dict), vectors=None, specials=SPECIAL_TOKENS)

        return glove.stoi, glove.itos







