# Imports
import numpy as np
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from DependencyParserModel import *
import tqdm
from utils.RunAndEvaluation import *
import matplotlib.pyplot as plt
from utils.DataPreprocessing import *

# Hyper Params
WORD_EMBEDDING_DIM = 100
TAG_EMBEDDING_DIM = 25
EPOCHS = 30
LEARNING_RATE = 0.01 # TODO
ACUMULATE_GRAD_STEPS = 5 # This is the actual batch_size, while we officially use batch_size=1
# uncomment for debugging
# CUDA_LAUNCH_BLOCKING = 1 #

def main():

    # Data paths
    data_dir = "Data/"
    # path_train = data_dir + "train.labeled"
    path_train = data_dir + "small_train.labeled"
    # path_test = data_dir + "test.labeled"
    path_test = data_dir + "small_test.labeled"

    # Data Preprocessing
    # Create Dictionaries of counts of words and tags from train + test
    word_dict, tag_dict = get_vocabs([path_train, path_test])

    # Prep Train Data
    train = DepDataset(word_dict,
                       tag_dict,
                       path_train,
                       word_embedding_dim=WORD_EMBEDDING_DIM,
                       padding=False) # TODO padding not in use
    train_dataloader = DataLoader(train, shuffle=True)

    # Prep Test Data
    test = DepDataset(word_dict,
                      tag_dict,
                      path_test,
                      word_embedding_dim=WORD_EMBEDDING_DIM,
                      padding=False) # TODO padding not in use
    test_dataloader = DataLoader(test, shuffle=False)

    # Dependency Parser Model
    word_vocab_size = len(train.word_to_idx_dict.keys())
    tag_vocab_size = len(train.tag_to_idx_dict.keys())
    model = KiperwasserDependencyParser(word_vocab_size,
                                        tag_vocab_size)

    # Determine if have GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    if use_cuda:
        print("using cuda")
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    print("Training Started")
    accuracy_train_list = []
    loss_train_list = []
    loss_test_list = []
    accuracy_test_list = []
    for epoch in range(1, EPOCHS+1):
        # Forward + Backward on train
        train_acc, train_loss = run_and_evaluate(model,
                                                 train_dataloader,
                                                 optimizer=optimizer,
                                                 acumelate_grad_steps=ACUMULATE_GRAD_STEPS)

        # Evaluate on test
        test_acc, test_loss = run_and_evaluate(model,
                                               test_dataloader,
                                               is_test=True)  # uses chu-liu

        # Statistics for plots
        loss_train_list.append(train_loss)
        loss_test_list.append(test_acc)
        accuracy_train_list.append(train_acc)
        accuracy_test_list.append(test_acc)

        print(f"Epoch {epoch} Completed,\tCurr Train Loss {train_loss}\t"
              f"Curr Train Accuracy: {train_acc}\t Curr Test Accuracy: {test_acc}\t"
              f"Curr Test Loss: {test_loss}\n")

        # Plot Accuracy
        create_graph(accuracy_train_list, accuracy_test_list, "Accuracy")

        # Plot Loss
        create_graph(loss_train_list, loss_test_list, "Loss")

        # Save model
        torch.save(model, "models/model{}.pth".format(epoch))


if __name__ == "__main__":
    main()

