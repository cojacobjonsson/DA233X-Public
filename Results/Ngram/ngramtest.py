from nltk import ngrams
from nltk.tokenize import word_tokenize
import time

models = ['../Data/cot130.txt', '../Data/cot5080.txt', '../Data/mle.txt', '../Data/seqgan5080.txt']
test_set = '../Data/test.txt'

def lines(filename):
    lst = []
    with open(filename) as f:
        lst = [line for line in f.read().split('\n') if line.strip() != ''] #f.read().splitlines()
    return lst

def get_ngrams(lst, n=2):
    ngramset = set()
    for sent in lst:
        for ngram in ngrams(word_tokenize(sent), n):
            ngramset.add(ngram)
    return ngramset

def get_N_ngrams(lst, n=2):
    sent_lengths = [len(sent.split()) for sent in lst]
    N_n_grams = [x + 1 - n for x in sent_lengths]
    return sum(N_n_grams)

def BLEU(models, test_set, n=2):
    test = lines(test_set)
    ref_ngrams = get_ngrams(test)

    for model in models:
        model_dataset = lines(model)
        model_ngrams = get_ngrams(model_dataset)

        intersect = ref_ngrams.intersection(model_ngrams)

        print('model: ' + model + '\tBLEU' + str(n) +': ' + str(len(intersect)/len(model_ngrams)))

def selfBLEU(models, n=2):
     for model in models:
        model_dataset = lines(model)
        self_bleu = 0
        tic = time.time()
        for i, sent in enumerate(model_dataset[:100]): # too slow otherwise
            sent_ngrams = get_ngrams([sent])
            #print(sent_ngrams)
            model_dataset.pop(i)
            corpus_ngrams = get_ngrams(model_dataset)
            #print(corpus_ngrams)
            model_dataset.insert(i, sent)

            intersect = sent_ngrams.intersection(corpus_ngrams)
            if len(sent_ngrams) > 0:
                self_bleu += len(intersect)/len(sent_ngrams)

            if i%5 == 0: # for progress check
                print('time: ' + str(time.time() - tic))
                tic = time.time()
                print('i: ' + str(i) + ', selfBLEU' + str(n) +': ' + str(self_bleu))

        print('model: ' + model + '\tselfBLEU' + str(n) +': ' + str(self_bleu/len(model_dataset[:100])))

def unique_ngrams(models, n=2):
    for model in models:
        model_dataset = lines(model)
        ngrams = get_ngrams(model_dataset, n)
        N_ngrams = get_N_ngrams(model_dataset, n)

        print('model: ' + model + '\tunique ' + str(n) +'-grams: ' + str(100*len(ngrams)/N_ngrams) + '%')

#BLEU(models, test_set, 2)
#selfBLEU(models)
unique_ngrams(models, 3)
