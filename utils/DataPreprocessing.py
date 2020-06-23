from torch.utils.data.dataset import Dataset
from collections import Counter
from torchtext.vocab import Vocab
import torch

ROOT_WORD = ROOT_TAG = "<root>"
UNKNOWN_TOKEN = UNKNOWN_TAG = "<unk>"
SPECIAL_TOKENS = [ROOT_WORD, UNKNOWN_TOKEN]


# TODO min and max thresholds for word occurences
def get_vocabs(list_of_paths, lower_case):
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
                    curr_word = splited_words[1].lower() if lower_case else splited_words[1]
                    curr_tag = splited_words[3]

                    word_dict[curr_word] = word_dict.get(curr_word, 0) + 1
                    tag_dict[curr_tag] = tag_dict.get(curr_tag, 0) + 1

    return word_dict, tag_dict


class DepDataReader:
    # TODO comp use
    def __init__(self, file, word_dict, tag_dict, word_to_idx_dict, tag_to_idx_dict, comp=False, lower_case=True):
        self.file = file
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.sentences = []
        self.word_to_idx_dict = word_to_idx_dict
        self.tag_to_idx_dict = tag_to_idx_dict
        self.comp = comp
        self.lower_case = lower_case
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
                    curr_word = splited_words[1].lower() if self.lower_case else splited_words[1]
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
    # TODO comp use
    def __init__(self, word_dict, tag_dict, file_path: str, word_embedding_name_or_size, comp, min_freq, lower_case):
        super().__init__()
        self.file = file_path
        self.word_vocab_size = len(word_dict)  # number of words in vocabulary
        # TODO delete self.word_vectors
        self.word_to_idx_dict, self.words_list, self.word_vectors = DepDataset.init_word_embeddings(word_dict,
                                                                                                    word_embedding_name_or_size,
                                                                                                    min_freq)
        self.tag_to_idx_dict, self.tags_list = DepDataset.init_tag_vocab(tag_dict)
        # TODO delete self.unknown_idx,root_idx
        self.unknown_idx = self.word_to_idx_dict.get(UNKNOWN_TOKEN)
        self.unknown_tag_idx = self.tag_to_idx_dict.get(UNKNOWN_TAG)
        self.root_idx = self.word_to_idx_dict.get(ROOT_WORD)
        datareader = DepDataReader(self.file, word_dict, tag_dict, self.word_to_idx_dict, self.tag_to_idx_dict, comp, lower_case)
        self.sentences_dataset = datareader.sentences
        sentences_lens = [len(sentence[0]) - 1 for sentence in self.sentences_dataset]
        self.num_edges = sum(sentences_lens)  # num of words
        self.num_sentences = len(self.sentences_dataset)

    def __len__(self):
        """returns num of sentences in data"""
        return self.num_sentences

    def __getitem__(self, index):
        word_embed_idx, tag_embed_idx, head_indexes_in_sentence = self.sentences_dataset[index]
        return word_embed_idx, tag_embed_idx, head_indexes_in_sentence

    @staticmethod
    def init_word_embeddings(word_dict, word_embedding_name_or_size, min_freq):
        if not word_embedding_name_or_size.isdigit():
            glove = Vocab(Counter(word_dict), vectors=word_embedding_name_or_size, specials=SPECIAL_TOKENS,
                          min_freq=min_freq)
        else:
            glove = Vocab(Counter(word_dict), vectors=None, specials=SPECIAL_TOKENS, min_freq=min_freq)

        return glove.stoi, glove.itos, glove.vectors

    @staticmethod
    def init_tag_vocab(tag_dict):
        glove = Vocab(Counter(tag_dict), vectors=None, specials=SPECIAL_TOKENS)

        return glove.stoi, glove.itos
