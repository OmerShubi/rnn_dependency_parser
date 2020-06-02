# Imports
import numpy as np
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from DependencyParserModel import *
import tqdm
from utils.evaluation import *
import matplotlib.pyplot as plt
from utils.getVocabulary import *

# Hyper Params
WORD_EMBEDDING_DIM = 100
TAG_EMBEDDING_DIM = 25
EPOCHS = 20
LEARNING_RATE = 0.01

# Data paths
data_dir = "Data/"
path_train = data_dir + "train.labeled"
path_test = data_dir + "test.labeled"
paths_list = [path_train, path_test]

word_dict, pos_dict = get_vocabs(paths_list)

# Prep Train Data
train = DepDataset(word_dict, pos_dict, path_train, word_embedding_dim=WORD_EMBEDDING_DIM, padding=False) # TODO padding not in use
train_dataloader = DataLoader(train, shuffle=True)

# Prep Test Data
test = DepDataset(word_dict, pos_dict, path_test, word_embedding_dim=WORD_EMBEDDING_DIM, padding=False) # TODO padding not in use
test_dataloader = DataLoader(test, shuffle=False)

# Dependency Parser Model
model = KiperwasserDependencyParser(len(train.word_idx_mappings.keys()),
                                    len(train.pos_idx_mappings.keys()),
                                    tag_embedding_dim=TAG_EMBEDDING_DIM,
                                    word_embedding_dim=WORD_EMBEDDING_DIM,
                                    word_embeddings=None)

# TRAIN
# CUDA_LAUNCH_BLOCKING=1 # TODO ?

word_vocab_size = len(train.word_idx_mappings)
tag_vocab_size = len(train.pos_idx_mappings)

# Determine if have GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
if use_cuda:
    print("using cuda")
    model.cuda()


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
acumulate_grad_steps = 5  # This is the actual batch_size, while we officially use batch_size=1

# Training start
print("Training Started")
accuracy_list = []
loss_list = []
for epoch in range(1, EPOCHS+1):
    total_acc = 0  # to keep track of accuracy
    total_loss = 0  # To keep track of the loss value
    total_len = 0  #
    i = 0
    for batch_idx, input_data in enumerate(tqdm.tqdm(train_dataloader)):
        i += 1

        loss, dep_tree = model(tuple(input_data))
        loss = loss / acumulate_grad_steps
        loss.backward()

        if i % acumulate_grad_steps == 0:
            optimizer.step()
            model.zero_grad()
            print(f"Epoch {epoch} milestone,\tLoss {total_loss / (i + 1)}\tAccuracy: {total_acc / total_len}")

        total_loss += loss.item()

        # TODO change to number of correct
        curr_len = len(dep_tree[1:])
        acc = acc_one_sen(dep_tree, input_data[2])
        total_acc += curr_len * acc
        total_len += curr_len

    total_loss = total_loss / len(train)
    total_acc = total_acc / total_len
    loss_list.append(float(total_loss))
    accuracy_list.append(float(total_acc))
    test_acc = evaluate(model, test_dataloader)  # uses chu-liu
    e_interval = i
    print(f"Epoch {epoch} Completed,\tLoss {np.mean(loss_list[-e_interval:])}\t"
          f"Accuracy: {np.mean(accuracy_list[-e_interval:])}\t Test Accuracy: {test_acc}")

    # Plot Accuracy
    plt.figure()
    plt.plot(accuracy_list, c="red", label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('../Graphs/Acc.png')

    # Plot Loss
    plt.figure()
    plt.plot(loss_list, c="blue", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('../Graphs/Loss.png')

    # Save model
    torch.save(model, "../models/model{}.pth".format(epoch))


# if __name__ == "__main__":
#     main()

