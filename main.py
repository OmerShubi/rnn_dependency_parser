# Imports
import numpy as np
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from DependencyParserModel import *
import tqdm
from utils.evaluation import *
import matplotlib.pyplot as plt
from utils.DataPreprocessing import *

# Hyper Params
WORD_EMBEDDING_DIM = 100
TAG_EMBEDDING_DIM = 25
EPOCHS = 5
LEARNING_RATE = 0.01
ACUMULATE_GRAD_STEPS = 3 # This is the actual batch_size, while we officially use batch_size=1


def main():

    # Data paths
    data_dir = "Data/"
    # path_train = data_dir + "train.labeled"
    path_train = data_dir + "small.labeled"
    # path_test = data_dir + "test.labeled"
    path_test = data_dir + "small.labeled"
    paths_list = [path_train, path_test]

    # Data Preprocessing
    # Create Dictionaries of counts of words and tags from train + test
    word_dict, pos_dict = get_vocabs(paths_list)

    # Prep Train Data
    train = DepDataset(word_dict, pos_dict, path_train, word_embedding_dim=WORD_EMBEDDING_DIM, padding=False) # TODO padding not in use
    train_dataloader = DataLoader(train, shuffle=True)

    # Prep Test Data
    test = DepDataset(word_dict, pos_dict, path_test, word_embedding_dim=WORD_EMBEDDING_DIM, padding=False) # TODO padding not in use
    test_dataloader = DataLoader(test, shuffle=False)

    # Dependency Parser Model
    word_vocab_size = len(train.word_idx_mappings.keys())
    pos_vocab_size = len(train.pos_idx_mappings.keys())
    model = KiperwasserDependencyParser(word_vocab_size,
                                        pos_vocab_size,
                                        tag_embedding_dim=TAG_EMBEDDING_DIM,
                                        word_embedding_dim=WORD_EMBEDDING_DIM,
                                        word_embeddings=None)

    # TRAIN
    # CUDA_LAUNCH_BLOCKING=1 # TODO ?

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
    accuracy_list = []
    loss_list = []
    for epoch in range(1, EPOCHS+1):
        total_acc = 0  # to keep track of accuracy
        total_loss = 0  # To keep track of the loss value
        total_len = 0
        i = 0
        for batch_idx, input_data in enumerate(tqdm.tqdm(train_dataloader)):
            i += 1

            loss, dep_tree = model(tuple(input_data))
            loss = loss / ACUMULATE_GRAD_STEPS
            loss.backward()

            if i % ACUMULATE_GRAD_STEPS == 0:
                optimizer.step() # updates params
                model.zero_grad()
                print(f"Epoch {epoch} milestone,\tLoss {total_loss / (i + 1)}\tAccuracy: {total_acc / total_len}")

            total_loss += loss.item()
            curr_len = len(dep_tree[1:])
            true_heads = input_data[2]
            total_acc += num_of_correct_one_sen(dep_tree, true_heads)
            total_len += curr_len

        # Statistics for plots
        total_loss = total_loss / len(train)
        total_acc = total_acc / total_len
        loss_list.append(float(total_loss))
        accuracy_list.append(float(total_acc))

        # Evaluate on test
        test_acc, _ = evaluate(model, test_dataloader)  # uses chu-liu

        print(f"Epoch {epoch} Completed,\tCurr Loss {total_loss}\t"
              f"Curr Train Accuracy: {total_acc}\t Curr Test Accuracy: {test_acc}\n")

        # Plot Accuracy
        plt.figure()
        plt.plot(accuracy_list, c="red", label="Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('Graphs/Acc.png')

        # Plot Loss
        plt.figure()
        plt.plot(loss_list, c="blue", label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('Graphs/Loss.png')

        # Save model
        torch.save(model, "models/model{}.pth".format(epoch))


if __name__ == "__main__":
    main()

