import numpy as np

def read_lines(file_name):
    file_loc = file_name
    sents = []
    with open(file_loc, 'r') as f:
        for line in f:
            sents.append(line.strip())
    return sents

def sent_stats(sents, lim=40):
    num_sents = len(sents)
    sent_lengths = [len(sent.split()) for sent in sents]
    too_long = sum(sent > lim for sent in sent_lengths)
    return num_sents, min(sent_lengths), max(sent_lengths), np.round(np.mean(sent_lengths), 2), too_long

def print_stats(file_name):
    sents = read_lines(file_name)
    num_sents, min_len, max_len, mean_len, too_long = sent_stats(sents)
    print('\n **** Stats for file: ' + file_name + ' ****')
    print('# sents: ' + str(num_sents))
    print('Min length: ' + str(min_len))
    print('Max length: ' + str(max_len))
    print('Mean length: ' + str(mean_len))
    print('Too long: ' + str(too_long))

coco = 'Texygen/image_coco.txt'
emnlp = 'Texygen/emnlp_news.txt'

test_coco = 'Texygen/testdata/test_coco.txt'
test_emnlp = 'Texygen/testdata/test_emnlp.txt'

data = 'Project/full_dataset_final_adj2.txt'
'''
print_stats(data)
print_stats(coco)
print_stats(emnlp)
print_stats(test_coco)
print_stats(test_emnlp)
'''

models = ['../Results/Data/cot130.txt', '../Results/Data/cot5080.txt', '../Results/Data/mle.txt', '../Results/Data/seqgan5080.txt']

for model in models:
    print_stats(model)