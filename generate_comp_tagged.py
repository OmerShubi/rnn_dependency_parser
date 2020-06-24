from torch.utils.data.dataloader import DataLoader
import DependencyParserModel
from utils.RunAndEvaluation import *
from torch import load
from utils.DataPreprocessing import *


def main():
    comp_path = "Data/comp.unlabeled"

    # Expects pretrained models to be stored under 'models' dir
    # Expects input data to be stored under 'Data' dir
    # Expects last char before ending to be 1/2 - model_id
    path_model1 = "models/model1.pth"
    path_model2 = "models/model2.pth"
    models_paths = [path_model1, path_model2]

    """
    train = DepDataset(word_dict=word_dict,
                       tag_dict=tag_dict,
                       file_path=path_train,
                       word_embedding_name_or_size=word_embedding_name_or_size,
                       comp=args.comp,
                       min_freq=params_dict["min_freq"],
                       lower_case=lower_case)"""

    for model_path in models_paths:
        model_id = model_path.split(".")[0][-1]
        comp_tagged_path = f"comp_m{model_id}_206348187.labeled"  # TODO change name by req

        # load model
        model = load(model_path)

        # Get the dictionaries that the model trained on
        word_dict, tag_dict = model.word_dict, model.tag_dict  # TODO maybe use word_to_index_dict

        # Preprocess the competition file
        comp = DepDataset(word_dict=word_dict,
                          tag_dict=tag_dict,
                          file_path=comp_path,
                          word_embedding_name_or_size=model.word_embedding_dim,
                          min_freq=1,
                          lower_case=True,  # TODO
                          comp=True)
        comp_dataloader = DataLoader(comp, shuffle=False)

        # Infer the competition file
        new_sentences = comp_infer(model, comp_dataloader)

        # Write prediction to file
        comp_writer(new_sentences, comp_tagged_path, comp_path)


def comp_infer(model, comp_dataloader):
    # Extract index sentences
    sentences = comp_dataloader.dataset.sentences_dataset

    for batch_idx, input_data in enumerate(tqdm.tqdm(comp_dataloader)):
        # Predict the tree structure
        predicted_tree_heads = model.infer(tuple(input_data), is_test=True, is_comp=True)

        # Bundle the word, tag and infered heads back together
        sentences[batch_idx] = input_data[0], input_data[1], predicted_tree_heads
    return sentences


def comp_writer(sentences, file_path, original_file_path):
    # Extract the input (not inferenced) data
    with open(original_file_path, "r") as f:
        originial_data = f.readlines()

    num_lines = 0
    with open(file_path, "w") as f:
        for sentence in sentences:
            _, _, heads = sentence
            # heads.size(0) num of words in sentence
            for i in range(1, heads.size(0)):
                original_sentence = originial_data[num_lines].split("\t")

                # rebuild the line with the predicted head
                f.write(get_line(i, original_sentence[1], original_sentence[3], heads[i]))
                num_lines += 1
            f.write("\n")
            num_lines += 1


def get_line(num, word, pos, head):
    return f"{num}\t{word}\t_\t{pos}\t_\t_\t{head}\t_\t_\t_\n"


if __name__ == '__main__':
    main()
