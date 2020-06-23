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
def run_and_evaluate(model, dataloader, accumulate_grad_steps=None, optimizer=None, is_test=False):
    total_acc,  total_loss = 0, 0
    i = 0
    # torch.no_grad - temp covert all grad flags to False
    cm = torch.no_grad() if is_test else nullcontext()
    if is_test:
        model.eval()
    else:
        model.train()
    with cm:
        for batch_idx, input_data in enumerate(tqdm.tqdm(dataloader)):
            loss, predicted_tree_heads = model(tuple(input_data))
            if not is_test:
                i += 1
                loss = loss / accumulate_grad_steps
                loss.backward()
                if i % accumulate_grad_steps == 0:
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

def create_graph(train_list, test_list, label, time):
    sns.set()
    color = "red" if label == "Accuracy" else "blue"
    plt.figure()
    plt.plot(train_list, c=color, label=label + '_train', linestyle='-')
    plt.plot(test_list, c=color, label=label + '_test',  linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f'Graphs/{label}_{time}.png')
    plt.close()