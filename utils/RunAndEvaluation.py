import csv
from contextlib import nullcontext
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm

matplotlib.use('Agg')


def run_and_evaluate(model, dataloader, accumulate_grad_steps=None, optimizer=None, is_test=False):
    total_acc, total_loss = 0, 0
    i = 0
    # torch.no_grad - temp covert all grad flags to False
    cm = torch.no_grad() if is_test else nullcontext()
    if is_test:
        model.eval()
    else:
        model.train()
    with cm:
        for batch_idx, input_data in enumerate(tqdm.tqdm(dataloader)):
            loss, predicted_tree_heads = model(tuple(input_data), is_test=is_test)
            if not is_test:
                i += 1
                loss = loss / accumulate_grad_steps
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                if i % accumulate_grad_steps == 0:
                    optimizer.step()  # updates params
                    model.zero_grad()

            total_acc += num_of_correct_one_sen(predicted_tree_heads, input_data[2])  # input_data[2] = true_heads
            total_loss += loss.item()

        total_acc = total_acc / dataloader.dataset.num_edges
        total_loss = total_loss / len(dataloader)

    return total_acc, total_loss


def num_of_correct_one_sen(pred_heads, true_heads):
    pred_heads = pred_heads[1:]  # remove root head (-1)
    true_heads = true_heads[0][1:]  # remove root head (-1)
    return float((pred_heads.eq(true_heads)).sum())


def create_graph(train_list, test_list, label, time):
    sns.set()
    color = "red" if label == "Accuracy" else "blue"
    plt.figure()
    plt.plot(train_list, c=color, label=label + '_train', linestyle='-')
    plt.plot(test_list, c=color, label=label + '_test', linestyle='--')
    plt.title(f"{label} of train and test for each epoch")
    plt.xlabel("Epochs")
    plt.ylabel(f"{label} Value")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f'Graphs/{label}_{time}.png')
    plt.close()


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
            [args.debug, params_dict['num_epochs'], np.argmax(np.array(accuracy_test_list)) + 1, max_accur, args.msg,
             start_time_printable] +
            list(params_dict.values()))