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
from ax import optimize  # TODO

# uncomment for debugging
# CUDA_LAUNCH_BLOCKING = 1 #

parameters_basic_model = {"accumulate_grad_step": 5,
                          "optimizer_method": "{'optim': optim.Adam, 'lr': 0.001}",
                          "lstm_hidden_dim": 0,
                          "word_embedding_name_or_size_and_freeze_flag": "('100', False)",
                          "tag_embedding_dim": 25,
                          "mlp_hidden_dim": 100,
                          "bilstm_layers": 2,
                          "dropout_alpha": 0.25,
                          "lstm_dropout": 0.0,
                          "activation": "nn.Tanh",
                          "min_freq": 1,
                          'mlp_dropout': 0.0,
                          }  # Todo lowercase flag

parameters_advanced_model = {"accumulate_grad_step": 15,
                             "optimizer_method": "{'optim': optim.Adam, 'lr': 0.003}",
                             "lstm_hidden_dim": 0,
                             "word_embedding_name_or_size_and_freeze_flag": "('100', False)",
                             "tag_embedding_dim": 25,
                             "mlp_hidden_dim": 150,
                             "bilstm_layers": 3,
                             "dropout_alpha": 0.1,
                             "lstm_dropout": 0.0,
                             "activation": "nn.ReLU",
                             "min_freq": 1,
                             'mlp_dropout': 0.3,
                             }


def optimization_wrapper(args, logger, path_train, path_test, params_dict, lower_case_flag=True):
    """
    TODO
    :param args:
    :param logger:
    :param path_train:
    :param path_test:
    :param params_dict:
    :param lower_case_flag:
    :return:
    """

    logger.debug(f"Using params:{params_dict.items()}")

    word_embedding_name_or_size, freeze_embedding = eval(params_dict["word_embedding_name_or_size_and_freeze_flag"])

    lower_case = False if word_embedding_name_or_size == 'glove.840B.300d' else lower_case_flag

    # Data Preprocessing
    # Create Dictionaries of counts of words and tags from train + test
    # word_dict, tag_dict = get_vocabs([path_train, path_test]) # TODO delete, combine train and test files for competition
    word_dict, tag_dict = get_vocabs([path_train], lower_case)

    # Prep Train Data
    train = DepDataset(word_dict=word_dict,
                       tag_dict=tag_dict,
                       file_path=path_train,
                       word_embedding_name_or_size=word_embedding_name_or_size,
                       comp=args.comp,
                       min_freq=params_dict["min_freq"],
                       lower_case=lower_case)
    train_dataloader = DataLoader(train, shuffle=True)

    # Prep Test Data
    test = DepDataset(word_dict=word_dict,
                      tag_dict=tag_dict,
                      file_path=path_test,
                      word_embedding_name_or_size=word_embedding_name_or_size,
                      comp=args.comp,
                      min_freq=1,
                      lower_case=lower_case)
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
                                        lstm_dropout=params_dict['lstm_dropout'],
                                        mlp_dropout=params_dict['mlp_dropout'])

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
        start_time_printable = datetime.datetime.now().strftime('%d-%m-%H%M')

        accuracy_train_list, loss_train_list, loss_test_list, accuracy_test_list = [], [], [], []
        max_test_acc, prev_test_acc = -1, -1

        for epoch in range(1, args.num_epochs + 1):
            # Forward + Backward on train
            _, _ = run_and_evaluate(model,
                                    train_dataloader,
                                    accumulate_grad_steps=params_dict["accumulate_grad_step"],
                                    optimizer=optimizer)

            # # Evaluate on train
            train_acc, train_loss = run_and_evaluate(model,
                                                     train_dataloader,
                                                     is_test=True)

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

            # Store the best modek
            if test_acc > max_test_acc:
                max_test_acc = test_acc

                # Save model # TODO
                torch.save(model, f"models/model1_{start_time_printable}_{round(max_test_acc, 4)}.pth")

            if test_acc > prev_test_acc:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            prev_test_acc = test_acc

            if epochs_no_improve == args.n_epochs_stop:
                logger.debug(f'Early stopping after epoch {epoch} with max test acc of {max_test_acc}')
                break

        end_time = datetime.datetime.now().strftime('%d-%m-%H%M')

        # Save model
        torch.save(model, f"models/model1_{start_time_printable}.pth")

        # Plot Accuracy
        create_graph(accuracy_train_list, accuracy_test_list, "Accuracy", start_time_printable)

        # Plot Loss
        create_graph(loss_train_list, loss_test_list, "Loss", start_time_printable)

        logger.debug(f"training took {round(time.time() - start_time, 0)} seconds with max test acc of {max_test_acc}")

        # save results in csv
        write_results(accuracy_test_list, args, params_dict, start_time_printable)

        return max(accuracy_test_list)


def write_results(accuracy_test_list, args, params_dict, start_time_printable):
    """

    :param accuracy_test_list:
    :param args:
    :param params_dict:
    :param start_time_printable:
    :return:
    """
    # save results in csv
    with open("results/results.csv", 'a') as csv_file:
        writer = csv.writer(csv_file)
        max_accur = max(accuracy_test_list)
        writer.writerow(
            [args.debug, args.num_epochs, np.argmax(np.array(accuracy_test_list)) + 1, max_accur, args.msg,
             start_time_printable] +
            list(params_dict.values()))


def main():
    # TODO model 1 numexpochs 30, model 2 - maxxxx
    parser = ArgumentParser()
    parser.add_argument('--skip_train', help='if skip train', action='store_true', default=False)
    parser.add_argument('--model_path', help='if skip train - path model to load', type=str,
                        default="models/model1.pth")
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--msg', help='msg to write in log file', type=str, default='')
    parser.add_argument('--n_epochs_stop', help='early stopping in training', type=int, default=10)
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
                    "values": [5, 10, 25, 50, 150],
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
                {
                    "name": "mlp_dropout",
                    "type": "choice",
                    "is_numeric": False,
                    "values": [0.0, 0.15, 0.3]
                },
            ],
            evaluation_function=lambda p: optimization_wrapper(args, logger, path_train, path_test, p),
            minimize=False,
            total_trials=args.total_trails,
            objective_name="UAS accuracy",
        )

        logger.debug(f"{best_parameters},{best_values[0]}")
    else:
        optimization_wrapper(args, logger, path_train, path_test, parameters_advanced_model)


if __name__ == "__main__":
    main()
