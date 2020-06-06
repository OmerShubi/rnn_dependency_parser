import numpy as np
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import DependencyParserModel
import tqdm
from utils.RunAndEvaluation import *
import matplotlib.pyplot as plt
from torch import load
from utils.DataPreprocessing import *


TAG_EMBEDDING_DIM = 25
WORD_EMBEDDING_DIM = 100
EPOCHS = 5
LR = 0.01
ARTIFICIAL_BATCH = 50
# ARTIFICIAL_BATCH = 5

def main():
    path_model1 = "models/model1.pth"
    # path_model2 = "models/model2.pth"
    model = load(path_model1)
    word_dict, tag_dict = model.word_dict, model.tag_dict
    path_comp = "Data/comp.unlabeled"
    tagged_path_comp = "comp_m{}_206348187.labeled" # TODO change name by req
    comp = DepDataset(word_dict,
                      tag_dict,
                      path_comp,
                      comp=True)
    comp_dataloader = DataLoader(comp, shuffle=False)
    new_sentences = comp_infer(model, comp_dataloader)
    comp_writer(new_sentences, tagged_path_comp)

def comp_infer(model, comp_dataloader):
    sentences = comp_dataloader.dataset.sentences_dataset
    for batch_idx, input_data in enumerate(tqdm.tqdm(comp_dataloader)):
        predicted_tree_heads = model.infer(tuple(input_data), is_comp=True)
        sentences[batch_idx] = input_data[0], input_data[1], predicted_tree_heads
    return sentences

def comp_writer(sentences, file_path):
    with open(file_path, "w") as f:
        for sentence in sentences:
            words, tags, heads = sentence
            # heads.size(0) num of words in sentence
            for i in range(1, words.size(0)):
                f.write(get_line(i, words[i], tags[i], heads[i]))
            f.write("\n")


def get_line(num, word, pos, head):
    return f"{num}\t{word}\t_\t{pos}\t_\t_\t{head}\t_\t_\t_\n"


if __name__ == '__main__':
    main()
