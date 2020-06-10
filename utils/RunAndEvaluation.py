import datetime

import torch
import tqdm
import os
import matplotlib
import matplotlib.pyplot as plt
from contextlib import nullcontext
import seaborn as sns

matplotlib.use('Agg')


# TODO generate comp loss
def run_and_evaluate(model, dataloader, optimizer=None, acumelate_grad_steps=None, is_test = False):
    total_acc,  total_loss = 0, 0
    i = 0
    # torch.no_grad - temp covert all grad flags to False
    cm = torch.no_grad() if is_test else nullcontext()
    with cm:
        for batch_idx, input_data in enumerate(tqdm.tqdm(dataloader)):
            loss, predicted_tree_heads = model(tuple(input_data))
            if not is_test:
                i += 1
                loss = loss / acumelate_grad_steps
                loss.backward()
                if i % acumelate_grad_steps == 0:
                    optimizer.step()  # updates params
                    model.zero_grad()

            total_acc += num_of_correct_one_sen(predicted_tree_heads, input_data[2]) # input_data[2] = true_heads
            total_loss += loss.item()

        total_acc = total_acc / dataloader.dataset.num_edges
        total_loss = total_loss / len(dataloader)
    return total_acc, total_loss

# TODO generate comp tag - change from acc to total num calc
def num_of_correct_one_sen(pred_heads, true_heads):
    pred_heads = pred_heads[1:] # remove root head (-1)
    true_heads = true_heads[0][1:] # remove root head (-1)
    return float((pred_heads.eq(true_heads)).sum())

def create_graph(train_list, test_list, label):
    sns.set()
    color = "red" if label == "Accuracy" else "blue"
    plt.figure()
    plt.plot(train_list, c=color, label=label + '_train', linestyle='-')
    plt.plot(test_list, c=color, label=label + '_test',  linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f'Graphs/{label}_{datetime.datetime.now().strftime("%d-%m-%H%M")}.png')
    plt.close()


def run_all_models(models_dir,test_dataloader):
    acc_list = []
    loss_list = []
    for filename in sorted(os.listdir(models_dir)):
        filepath = models_dir + os.sep + filename
        if filepath.endswith(".pth"):
            print("evaluating model saved in:{}".format(filepath))
            model = torch.load(filepath)
            curr_acc, curr_loss = run_and_evaluate(model, test_dataloader)
            acc_list.append(curr_acc)
            loss_list.append(curr_loss)
    return acc_list, loss_list

def generate_test_graphs(models_dir,test_dataloader,graphs_dir,prefix = ""):
    acc_list, loss_list = run_all_models(models_dir, test_dataloader)
    plt.figure()
    plt.plot(acc_list, c="red", label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Test accuracy graph")
    plt.legend()
    plt.savefig(graphs_dir + os.sep + prefix + 'Acc.png')

    plt.figure()
    plt.plot(loss_list, c="blue", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Test loss graph")
    plt.legend()
    plt.savefig(graphs_dir + os.sep + prefix +'Loss.png')
