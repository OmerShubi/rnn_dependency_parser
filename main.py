# Imports
import torch.optim as optim
from torch import load
from torch.utils.data import DataLoader
import logging.config
from DependencyParserModel import *
from utils.DataPreprocessing import *
from utils.RunAndEvaluation import *
from argparse import ArgumentParser
import csv
import numpy as np
import time
from ax import optimize

# uncomment for debugging
# CUDA_LAUNCH_BLOCKING = 1 #
N_EPOCHS_STOP = 500
# TODO
parameters_basic_model = {"learning_rate": 0.001,
                          "word_embedding_dim": 100,
                          "tag_embedding_dim": 25,
                          "accumulate_grad_step": 5,
                          "optimizer_method": "optim.Adam",
                          "lstm_hidden_dim": 125,
                          "word_embedding_name_or_size": "",
                          "mlp_hidden_dim": 100,
                          "bilstm_layers": 2,
                          "dropout_alpha": 0.25,
                          "activation": "nn.Tanh",
                          "min_freq": 1,
                          "freeze_embedding": False
                          }

parameters_advanced_model = {"accumulate_grad_step": 20,
                             "optimizer_method": "{'optim': optim.Adadelta, 'lr': 1.0}",
                             "lstm_hidden_dim": 0,
                             "word_embedding_name_or_size_and_freeze_flag": "('glove.840B.300d', True)",
                             "tag_embedding_dim": 50,
                             "mlp_hidden_dim": 500,
                             "bilstm_layers": 3,
                             "dropout_alpha": 0.1,
                             "lstm_dropout": 0.0,
                             "activation": "nn.ReLU",
                             "min_freq": 3,
                             }


def optimization_wrapper(args, logger, word_dict, tag_dict, path_train, path_test, params_dict):

    logger.debug(f"Using params:{params_dict.items()}")

    word_embedding_name_or_size, freeze_embedding = eval(params_dict["word_embedding_name_or_size_and_freeze_flag"])

    # Prep Train Data
    train = DepDataset(word_dict=word_dict,
                       tag_dict=tag_dict,
                       file_path=path_train,
                       word_embedding_name_or_size=word_embedding_name_or_size,
                       comp=args.comp,
                       min_freq=params_dict["min_freq"])
    train_dataloader = DataLoader(train, shuffle=True)

    # Prep Test Data
    test = DepDataset(word_dict=word_dict,
                      tag_dict=tag_dict,
                      file_path=path_test,
                      word_embedding_name_or_size=word_embedding_name_or_size,
                      comp=args.comp,
                      min_freq=params_dict["min_freq"])
    test_dataloader = DataLoader(test, shuffle=False)

    # Dependency Parser Model
    model = KiperwasserDependencyParser(word_dict=word_dict,
                                        tag_dict=tag_dict,
                                        word_list=train.words_list,
                                        tag_list=train.tags_list,
                                        tag_embedding_dim=params_dict["tag_embedding_dim"],
                                        word_embedding_dim=word_embedding_name_or_size,
                                        pretrained_embedding=train.word_vectors,
                                        lstm_hidden_dim=params_dict["lstm_hidden_dim"],
                                        mlp_hidden_dim=params_dict["mlp_hidden_dim"],
                                        bilstm_layers=params_dict["bilstm_layers"],
                                        dropout_alpha=params_dict["dropout_alpha"],
                                        activation=params_dict["activation"],
                                        freeze_embedding=freeze_embedding,
                                        lstm_dropout=params_dict['lstm_dropout'])

    # Determine if have GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.debug(device)
    if use_cuda:
        logger.debug("using cuda")
        model.cuda()

    optimizer_dict = eval(params_dict["optimizer_method"])
    try:
        betas = optimizer_dict['betas']
        optimizer = optimizer_dict['optim'](model.parameters(), lr=optimizer_dict['lr'], betas=betas)
    except KeyError:
        optimizer = optimizer_dict['optim'](model.parameters(), lr=optimizer_dict['lr'])

    # Training

    if args.skip_train:
        model = load(args.model_path)
        # Evaluate on test
        test_acc, test_loss = run_and_evaluate(model,
                                               test_dataloader,
                                               is_test=True,
                                               accumulate_grad_steps=args.acumelate_grad_steps)
        logger.debug("test acc, test loss:", test_acc, test_loss)

        return test_acc

    else:
        logger.debug("Training Started")
        start_time = time.time()
        accuracy_train_list = []
        loss_train_list = []
        loss_test_list = []
        accuracy_test_list = []
        max_test_accur = -1
        for epoch in range(1, args.num_epochs + 1):
            # Forward + Backward on train
            train_acc, train_loss = run_and_evaluate(model,
                                                     train_dataloader,
                                                     accumulate_grad_steps=params_dict["accumulate_grad_step"],
                                                     optimizer=optimizer)

            # # Evaluate on train
            # train_acc, train_loss = run_and_evaluate(model,
            #                                          train_dataloader,
            #                                          is_test=True)

            start_time_test = time.time()
            # Evaluate on test
            test_acc, test_loss = run_and_evaluate(model,
                                                   test_dataloader,
                                                   is_test=True)
            if epoch == 1:
                logger.debug(f"tagging test took {round(time.time() - start_time_test, 2)} seconds")

            # Statistics for plots
            loss_train_list.append(train_loss)
            loss_test_list.append(test_acc)
            accuracy_train_list.append(train_acc)
            accuracy_test_list.append(test_acc)

            logger.debug(f"Epoch {epoch} Completed,\tCurr Train Loss {train_loss}\t"
                         f"Curr Train Accuracy: {train_acc}\t Curr Test Loss: {test_loss}\t"
                         f"Curr Test Accuracy: {test_acc}\n")

            if test_acc > max_test_accur:
                epochs_no_improve = 0
                max_test_accur = test_acc
            else:
                epochs_no_improve += 1
            if epochs_no_improve == N_EPOCHS_STOP:
                logger.debug(f'Early stopping after epoch {epoch} with max test acc of {max_test_accur}')
                break

        end_time = datetime.datetime.now().strftime('%d-%m-%H%M')

        # Save model
        torch.save(model, f"models/model1_{end_time}.pth")

        # Plot Accuracy
        create_graph(accuracy_train_list, accuracy_test_list, "Accuracy", end_time)

        # Plot Loss
        create_graph(loss_train_list, loss_test_list, "Loss", end_time)

        logger.debug(f"training took {round(time.time() - start_time, 0)} seconds")

        # save results in csv
        with open("results/results.csv", 'a') as csv_file:
            writer = csv.writer(csv_file)
            max_accur = max(accuracy_test_list)
            writer.writerow(
                [args.debug, args.num_epochs, np.argmax(np.array(accuracy_test_list)) + 1, max_accur, args.msg,
                 end_time] +
                list(params_dict.values()))
        return max_accur


def main():
    parser = ArgumentParser()
    parser.add_argument('--skip_train', help='if skip train', action='store_true', default=False)
    parser.add_argument('--model_path', help='if skip train - path model to load', type=str,
                        default="models/model1.pth")
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--msg', help='msg to write in log file', type=str, default='')
    parser.add_argument('--comp', action='store_true', default=False)
    parser.add_argument('--total_trails', type=int, default=70)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--search_hyperparams', action='store_true', default=False)
    args = parser.parse_args()

    # Gets or creates a logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    # Data paths
    data_dir = "Data/"
    if args.debug:
        path_train = data_dir + "small_train.labeled"
        path_test = data_dir + "small_test.labeled"
    else:
        path_train = data_dir + "train.labeled"
        path_test = data_dir + "test.labeled"
    logger.debug(f"Starting train: {path_train} test:{path_test}")

    # Data Preprocessing
    # Create Dictionaries of counts of words and tags from train + test
    # word_dict, tag_dict = get_vocabs([path_train, path_test]) # TODO delete, combine train and test files for competition
    word_dict, tag_dict = get_vocabs([path_train])
    if args.search_hyperparams:
        best_parameters, best_values, experiment, model = optimize(
            parameters=[
                {
                    "name": "accumulate_grad_step",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [2, 5, 10, 20],
                },
                {
                    "name": "optimizer_method",
                    "type": "choice",
                    "is_numeric": False,
                    "values": ["{'optim':optim.Adam, 'lr':0.001, 'betas':(0.9, 0.999)}",
                               "{'optim':optim.Adam, 'lr':0.01, 'betas':(0.9, 0.9)}",
                               "{'optim':optim.Adam, 'lr':0.001, 'betas':(0.9, 0.9)}",
                               "{'optim':optim.Adam, 'lr':0.0005, 'betas':(0.9, 0.9)}",
                               "{'optim':optim.SGD, 'lr':0.01}",
                               "{'optim':optim.SGD, 'lr':0.1}",
                               "{'optim':optim.SGD, 'lr':0.001}",
                               "{'optim':optim.SGD, 'lr':1.0}",
                               "{'optim':optim.AdamW, 'lr':0.001, 'betas':(0.9, 0.999)}",
                               "{'optim':optim.AdamW, 'lr':0.002, 'betas':(0.9, 0.9)}",
                               "{'optim':optim.AdamW, 'lr':0.0005, 'betas':(0.9, 0.9)}",
                               "{'optim':optim.AdamW, 'lr':0.001, 'betas':(0.9, 0.9)}",
                               "{'optim':optim.Adadelta, 'lr':1}",
                               "{'optim':optim.Adadelta, 'lr':2}"],
                },
                {
                    "name": "lstm_hidden_dim",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [125, 200, 300, 400, 0],
                },
                {
                    "name": "word_embedding_name_or_size_and_freeze_flag",
                    "type": "choice",
                    "is_numeric": False,
                    "values": ["('glove.6B.50d', True)",
                               "('glove.6B.50d', False)",
                               "('glove.6B.100d', True)",
                               "('glove.6B.100d', False)",
                               "('glove.6B.200d', True)",
                               "('glove.6B.200d', False)",
                               "('glove.6B.300d', True)",
                               "('glove.6B.300d', False)",
                               "('glove.840B.300d', True)",
                               "('glove.840B.300d', False)",
                               "('fasttext.en.300d', True)",
                               "('fasttext.en.300d', False)",
                               "('25', False)",
                               "('50', False)",
                               "('100', False)",
                               "('200', False)",
                               "('300', False)"]
                },
                {
                    "name": "tag_embedding_dim",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [5, 10, 25, 50],
                },
                {
                    "name": "mlp_hidden_dim",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [50, 100, 200, 300, 400, 500],
                },
                {
                    "name": "bilstm_layers",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [1, 2, 3, 4]
                },
                {
                    "name": "dropout_alpha",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [0.0, 0.05, 0.1, 0.25]
                },
                {
                    "name": "lstm_dropout",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [0.0, 0.15, 0.3]
                },
                {
                    "name": "activation",
                    "type": "choice",
                    "is_numeric": False,
                    "values": ["nn.Tanh", "nn.ReLU", "nn.Sigmoid", "nn.LeakyReLU"]
                },
                {
                    "name": "min_freq",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [1, 2, 3, 5]
                },
            ],
            evaluation_function=lambda p: optimization_wrapper(args, logger, word_dict, tag_dict, path_train, path_test,
                                                               p),
            minimize=False,
            total_trials=args.total_trails,
            objective_name="UAS accuracy",
        )

        logger.debug(f"{best_parameters},{best_values[0]}")
    else:
        optimization_wrapper(args, logger, word_dict, tag_dict, path_train, path_test, parameters_advanced_model)


if __name__ == "__main__":
    main()
