# Imports
import torch.optim as optim
from torch import load
from torch.utils.data import DataLoader

from DependencyParserModel import *
from utils.DataPreprocessing import *
from utils.RunAndEvaluation import *

# Hyper Params
WORD_EMBEDDING_DIM = 100
TAG_EMBEDDING_DIM = 25
EPOCHS = 10
LEARNING_RATE = 0.01  # TODO
ACUMULATE_GRAD_STEPS = 5  # This is the actual batch_size, while we officially use batch_size=1
DEBUG = False
run_train = True
load_model_and_run = False
path_model = "models/model_epoch20.pth"


# uncomment for debugging
# CUDA_LAUNCH_BLOCKING = 1 #

def main():
    # Data paths
    data_dir = "Data/"
    if DEBUG:
        path_train = data_dir + "small_train.labeled"
        path_test = data_dir + "small_test.labeled"
    else:
        path_train = data_dir + "train.labeled"
        path_test = data_dir + "test.labeled"

    # Data Preprocessing
    # Create Dictionaries of counts of words and tags from train + test
    word_dict, tag_dict = get_vocabs([path_train, path_test])

    # Prep Train Data
    train = DepDataset(word_dict,
                       tag_dict,
                       path_train,
                       padding=False)  # TODO padding not in use
    train_dataloader = DataLoader(train, shuffle=True)

    # Prep Test Data
    test = DepDataset(word_dict,
                      tag_dict,
                      path_test,
                      padding=False)  # TODO padding not in use
    test_dataloader = DataLoader(test, shuffle=False)

    # Dependency Parser Model
    model = KiperwasserDependencyParser(word_dict, tag_dict)

    # Determine if have GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    if use_cuda:
        print("using cuda")
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training

    if run_train:
        print("Training Started")
        accuracy_train_list = []
        loss_train_list = []
        loss_test_list = []
        accuracy_test_list = []
        for epoch in range(1, EPOCHS + 1):
            # Forward + Backward on train
            train_acc, train_loss = run_and_evaluate(model,
                                                     train_dataloader,
                                                     optimizer=optimizer,
                                                     acumelate_grad_steps=ACUMULATE_GRAD_STEPS)

            # Evaluate on test
            test_acc, test_loss = run_and_evaluate(model,
                                                   test_dataloader,
                                                   is_test=True)

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
            torch.save(model, f"models/model_epoch{epoch}.pth")

    if load_model_and_run:
        model = load(path_model)
        # Evaluate on test
        test_acc, test_loss = run_and_evaluate(model,
                                               test_dataloader,
                                               is_test=True)
        print("test acc, test loss:", test_acc, test_loss)


if __name__ == "__main__":
    main()
