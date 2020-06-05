import torch
import tqdm
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def evaluate(model, dataloader):
    # TODO generate comp loss
    acc = 0
    total_len = 0
    total_acc = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(tqdm.tqdm(dataloader)):
            loss, predicted_tree = model(tuple(input_data))
            curr_len = input_data[2][0].size(0)
            total_acc += num_of_correct_one_sen(predicted_tree, input_data[2])
            total_loss += curr_len * loss
            total_len += curr_len
        total_acc = total_acc / total_len
        total_loss = total_loss / total_len
    return total_acc, total_loss

# TODO generate comp tag- change grom acc to total num calc
def num_of_correct_one_sen(pred_heads, true_heads):
    pred_heads = pred_heads[1:] # remove root head (-1)
    true_heads = true_heads[0][1:] # remove root head (-1)
    return float((pred_heads.eq(true_heads)).sum())


def run_all_models(models_dir,test_dataloader):
    acc_list = []
    loss_list = []
    for filename in sorted(os.listdir(models_dir)):
        filepath = models_dir + os.sep + filename
        if filepath.endswith(".pth"):
            print("evaluating model saved in:{}".format(filepath))
            model=torch.load(filepath)
            curr_acc, curr_loss = evaluate(model, test_dataloader)
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
