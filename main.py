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

DEBUG = True

# uncomment for debugging
# CUDA_LAUNCH_BLOCKING = 1 #

def optimization_wrapper(args, logger, word_dict, tag_dict, path_train, path_test, learning_rate, word_embedding_dim,
                         tag_embedding_dim, accumulate_grad_step, optimizer_method):

    # Prep Train Data
    train = DepDataset(word_dict=word_dict,
                       tag_dict=tag_dict,
                       file_path=path_train,
                       pretrained_embedding=args.pretrained_embedding,
                       comp=args.comp)
    train_dataloader = DataLoader(train, shuffle=True)

    # Prep Test Data
    test = DepDataset(word_dict=word_dict,
                       tag_dict=tag_dict,
                       file_path=path_test,
                       pretrained_embedding=args.pretrained_embedding,
                       comp=args.comp)
    test_dataloader = DataLoader(test, shuffle=False)

    # Dependency Parser Model
    model = KiperwasserDependencyParser(word_dict=word_dict,
                                        tag_dict=tag_dict,
                                        word_list=train.words_list,
                                        tag_list=train.tags_list,
                                        tag_embedding_dim=tag_embedding_dim,
                                        word_embedding_dim=word_embedding_dim,
                                        lstm_out_dim=args.lstm_out_dim,
                                        pretrained_embedding=train.word_vectors,
                                        lstm_hidden_dim=args.lstm_hidden_dim,
                                        mlp_hidden_dim=args.mlp_hidden_dim,
                                        bilstm_layers=args.bilstm_layers,
                                        dropout_alpha=args.dropout_alpha)

    # Determine if have GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.debug(device)
    if use_cuda:
        logger.debug("using cuda")
        model.cuda()

    optimizer_method = eval(optimizer_method)
    optimizer = optimizer_method(model.parameters(), lr=learning_rate)

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
        for epoch in range(1, args.num_epochs + 1):
            # Forward + Backward on train
            _, _ = run_and_evaluate(model,
                                                     train_dataloader,
                                                     accumulate_grad_steps=accumulate_grad_step,
                                                     optimizer=optimizer)

            # Evaluate on train
            train_acc, train_loss = run_and_evaluate(model,
                                                     train_dataloader,
                                                     is_test=True)

            start_time_test = time.time()
            # Evaluate on test
            test_acc, test_loss = run_and_evaluate(model,
                                                   test_dataloader,
                                                   is_test=True)
            if epoch == args.num_epochs:
                logger.debug(f"tagging test took {round(time.time() - start_time_test, 2)} seconds")

            # Statistics for plots
            loss_train_list.append(train_loss)
            loss_test_list.append(test_acc)
            accuracy_train_list.append(train_acc)
            accuracy_test_list.append(test_acc)

            logger.debug(f"Epoch {epoch} Completed,\tCurr Train Loss {train_loss}\t"
                         f"Curr Train Accuracy: {train_acc}\t Curr Test Loss: {test_loss}\t"
                         f"Curr Test Accuracy: {test_acc}\n")

        end_time = datetime.datetime.now().strftime('%d-%m-%H%M')

        # Save model
        torch.save(model, f"models/model1_{end_time}.pth")

        # Plot Accuracy
        create_graph(accuracy_train_list, accuracy_test_list, "Accuracy", end_time)

        # Plot Loss
        create_graph(loss_train_list, loss_test_list, "Loss", end_time)

        logger.debug(f"training took {round(time.time()-start_time,0)} seconds")

        # save results in csv
        with open("results/results.csv", 'a') as csv_file:
            writer = csv.writer(csv_file)
            max_accur = max(accuracy_test_list)
            writer.writerow([args.num_epochs,np.argmax(np.array(accuracy_test_list)), max_accur, learning_rate,
                             accumulate_grad_step, tag_embedding_dim, word_embedding_dim, args.lstm_out_dim, args.lstm_hidden_dim,
                             args.pretrained_embedding, args.mlp_hidden_dim, args.bilstm_layers, args.dropout_alpha, optimizer_method,
                             args.msg, end_time])
        return max_accur

def main():
    parser = ArgumentParser()
    parser.add_argument('--skip_train', help='if skip train', action='store_true', default=False)
    parser.add_argument('--model_path', help='if skip train - path model to load', type=str,
                        default="models/model1.pth")
    parser.add_argument('--word_embedding_dim', type=int, default=100)
    parser.add_argument('--tag_embedding_dim', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=int, default=0.01)
    parser.add_argument('--accumulate_grad_step', type=int, default=5)
    parser.add_argument('--msg', help='msg to write in log file', type=str, default='')
    parser.add_argument('--comp',  action='store_true', default=False)
    parser.add_argument('--pretrained_embedding', default=None) # "glove.6B.100d"
    parser.add_argument('--dropout_alpha', default=None)
    parser.add_argument('--lstm_out_dim', default=None)
    parser.add_argument('--lstm_hidden_dim', default=None)
    parser.add_argument('--mlp_hidden_dim', type=int, default=100)
    parser.add_argument('--bilstm_layers', type=int, default=2)
    args = parser.parse_args()


    # Gets or creates a logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    # Data paths
    data_dir = "Data/"
    if DEBUG:
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

    # optimization_wrapper(args, logger, word_dict, tag_dict, path_train, path_test)

    best_parameters, best_values, experiment, model = optimize(
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.001, 0.01],
            },
            {
                "name": "word_embedding_dim",
                "type": "range",
                "value_type": "int",
                "bounds": [100, 1000],
                "log_scale": True
            },
            {
                "name": "tag_embedding_dim",
                "type": "range",
                "value_type": "int",
                "bounds": [10, 100],
                "log_scale": True
            },
            {
                "name": "accumulate_grad_step",
                "type": "range",
                "value_type": "int",
                "bounds": [1, 10],
            },
            {
                "name": "optimizer_method",
                "type": "choice",
                "is_numeric": False,
                "values": ["optim.Adam", "optim.SGD"],
            }
        ],
        evaluation_function=lambda p : optimization_wrapper(args, logger, word_dict, tag_dict, path_train, path_test, p["learning_rate"]
                                                            ,p["word_embedding_dim"], p["tag_embedding_dim"], p["accumulate_grad_step"],
                                                            p["optimizer_method"]),
        minimize=False,
    )

    print(best_parameters, best_values)

if __name__ == "__main__":
    main()
