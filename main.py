# Imports
import datetime
from torch import load
from torch.utils.data import DataLoader
import logging.config
from DependencyParserModel import *
from utils.DataPreprocessing import *
from utils.RunAndEvaluation import *
from argparse import ArgumentParser
import time
import torch.optim as optim
from torch import manual_seed
manual_seed(0)

# uncomment for debugging
# CUDA_LAUNCH_BLOCKING = 1 #
from utils.RunAndEvaluation import write_results


def optimization_wrapper(args, logger, path_train, path_test, params_dict):
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

    lower_case = False if word_embedding_name_or_size == 'glove.840B.300d' else params_dict['lower_case_flag']

    # Data Preprocessing
    # Create Dictionaries of counts of words and tags from train path
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
                                               is_test=True)
        logger.debug(f"test acc {test_acc}, test loss: {test_loss}")

        return test_acc

    else:
        logger.debug("Training Started")

        start_time = time.time()
        start_time_printable = datetime.datetime.now().strftime('%d-%m-%H%M')

        accuracy_train_list, loss_train_list, loss_test_list, accuracy_test_list = [], [], [], []
        max_test_acc, prev_test_acc = -1, -1

        for epoch in range(1, params_dict['num_epochs'] + 1):
            # Forward + Backward on train
            train_acc, train_loss = run_and_evaluate(model,
                                                     train_dataloader,
                                                     accumulate_grad_steps=params_dict["accumulate_grad_step"],
                                                     optimizer=optimizer)

            if epoch == 1:
                # Evaluate on train
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

            # Store the best model # TODO what is the best mode??
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                # Save model
                torch.save(model, f"models/model_{start_time_printable}_{args.model_id}.pth")

            if test_acc > prev_test_acc:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            prev_test_acc = test_acc

            if epochs_no_improve == args.n_epochs_stop:
                logger.debug(f'Early stopping after epoch {epoch} with max test acc of {max_test_acc}')
                break

            torch.save(model, f"models/model_{epoch}_{start_time_printable}_{test_acc}_{args.model_id}.pth")

        # Plot Accuracy
        create_graph(accuracy_train_list, accuracy_test_list, "Accuracy",
                     f'model_{start_time_printable}_{args.model_id}')

        # Plot Loss
        create_graph(loss_train_list, loss_test_list, "Loss", f'model_{start_time_printable}_{args.model_id}')

        logger.debug(f"training took {round(time.time() - start_time, 0)} seconds with max test acc of {max_test_acc}")

        # save results in csv
        write_results(accuracy_test_list, args, params_dict, start_time_printable)

        return max(accuracy_test_list)


def main():
    """
    model1 Run example:
        python main.py --model_id=1

    model2 Run example:
        python main.py --model_id=2

    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('--skip_train', help='if skip train', action='store_true', default=False)
    parser.add_argument('--model_path', help='if skip train - path model to load', type=str,
                        default="models/model1.pth")
    parser.add_argument('--msg', help='msg to write in log file', type=str, default='')
    parser.add_argument('--n_epochs_stop', help='early stopping in training', type=int, default=200)
    parser.add_argument('--model_id', help='should be 1 or 2', type=int, default=2)
    parser.add_argument('--comp', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--combined_train_data', action='store_true', default=False)
    parser.add_argument('--do_cv', action='store_true', default=False)
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

    if args.combined_train_data:
        path_train = data_dir + "combined.labeled"
        path_test = data_dir + "validation.labeled"

    logger.debug(f"Starting train: {path_train} test:{path_test}")

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
                              'num_epochs': 30,
                              'lower_case_flag': False,
                              }

    parameters_advanced_model = {"accumulate_grad_step": 5,
                                 "optimizer_method": "{'optim': optim.SGD, 'lr': 1.0}",
                                 "lstm_hidden_dim": 200,
                                 "word_embedding_name_or_size_and_freeze_flag": "('glove.6B.100d', False)",
                                 "tag_embedding_dim": 25,
                                 "mlp_hidden_dim": 500,
                                 "bilstm_layers": 2,
                                 "dropout_alpha": 0.3,
                                 "lstm_dropout": 0.0,
                                 "activation": "nn.ReLU",
                                 "min_freq": 3,
                                 'mlp_dropout': 0.15,
                                 'num_epochs': 200,
                                 'lower_case_flag': True,
                                 }

    if args.model_id == 1:
        params = parameters_basic_model
    elif args.model_id == 2:
        params = parameters_advanced_model
    else:
        print("model id must be 1 or 2")
        return

    if args.do_cv:
        for i in range(10):
            path_train = f"cv/combined{i}.labeled"
            path_test = f"cv/val{i}.labeled"

            logger.debug(f"Starting train: {path_train} test:{path_test}")
            optimization_wrapper(args=args,
                                 logger=logger,
                                 path_train=path_train,
                                 path_test=path_test,
                                 params_dict=params)

    else:
        optimization_wrapper(args=args,
                             logger=logger,
                             path_train=path_train,
                             path_test=path_test,
                             params_dict=params)


if __name__ == "__main__":
    main()
