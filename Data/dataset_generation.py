import os
import re
from random import shuffle

def main():
    directory_in_str = './Project/processed/'
    filename_out = './Project/full_dataset.txt'
    directory = os.fsencode(directory_in_str)
    dataset = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(directory_in_str + filename, 'r', encoding='UTF-8') as f:
            for line in f:
                if len(line.split()) > 2: # only add sentences longer than two words to the dataset
                    dataset.append(line.lower())

    dataset = list(set(dataset)) # remove duplicates
    with open(filename_out, 'w', encoding='UTF-8') as f_out:
        f_out.write(''.join(dataset))

def sort():
    with open('Project/full_dataset_adjusted.txt', 'r', encoding='UTF-8') as dataset:
        lines = dataset.readlines()
        sorted_lines = sorted(lines, key=len, reverse=True)
        with open('Project/full_dataset_adj_sorted.txt', 'w', encoding='UTF-8') as data:
            data.writelines(sorted_lines)

def split():
    with open('Project/full_dataset_final_adj.txt', 'r', encoding='UTF-8') as dataset:
        lines = dataset.readlines()
        # randomize indices to split
        n_samples = len(lines)
        indices = list(range(n_samples))
        shuffle(indices)
        with open('Project/train.txt', 'w', encoding='UTF-8') as train:
            train_data = [lines[i] for i in indices[:10000]]
            train.writelines(train_data)
        with open('Project/test.txt', 'w', encoding='UTF-8') as test:
            test_data = [lines[i] for i in indices[10000:]]
            test.writelines(test_data)

def create_eval():
    with open('Project/test.txt', 'r', encoding='UTF-8') as dataset:
        lines = dataset.readlines()
        # randomize indices to split
        n_samples = len(lines)
        indices = list(range(n_samples))
        shuffle(indices)
        with open('Project/human_eval_set.txt', 'w', encoding='UTF-8') as eval:
            eval_data = [lines[i] for i in indices[:100]]
            eval.writelines(eval_data)

if __name__=="__main__":
    # main()
    #split()
    # sort()
    create_eval()