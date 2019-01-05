# Original creator: rvinas@github, https://gist.github.com/rvinas/cf5c4c47456834d7fd4e3328858cffe2
# Modified by Jacob Jonsson, cojacobjonsson@github

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential, load_model
import numpy as np
import pickle

maxlen = 44
n_epochs = 3

def prepare_sentence(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                 maxlen=maxlen - 1,
                                 padding='pre')[0]  # Pads before each sequence
        x.append(x_padded)
        y.append(w)
    return x, y

train_data = 'data/test.txt' # because reverse model
test_data = 'data/train.txt'

test = []
train = []

with open(test_data) as test_file:
    test = test_file.read().splitlines()

with open(train_data) as train_file:
    train = train_file.read().splitlines()

full_dataset = test + train

tokenizer = Tokenizer()
tokenizer.fit_on_texts(full_dataset)
vocab = tokenizer.word_index

models = ['data/cot130.txt', 'data/cot5080.txt', 'data/mle.txt', 'data/seqgan5080.txt']

for train_set in models:
    train = []

    with open(train_set) as train_file:
        train = train_file.read().splitlines()

    seqs = tokenizer.texts_to_sequences(train)

    # Slide windows over each sentence
    #maxlen = max([len(seq) for seq in seqs])
    x = []
    y = []
    for seq in seqs:
        x_windows, y_windows = prepare_sentence(seq, maxlen)
        x += x_windows
        y += y_windows
    x = np.array(x)
    y = np.array(y) - 1
    y = np.eye(len(vocab))[y]  # One hot encoding

    # Define model
    model = Sequential()
    model.add(Embedding(input_dim=len(vocab) + 1,  # vocabulary size. Adding an
                                                # extra element for <PAD> word
                        output_dim=5,  # size of embeddings
                        input_length=maxlen - 1))  # length of the padded sequences
    model.add(LSTM(32))
    model.add(Dense(len(vocab), activation='softmax'))
    model.compile('rmsprop', 'categorical_crossentropy')

    # Train network
    model.fit(x, y, epochs=n_epochs)

    # saving
    model.save('models/lm_' + str(train_set) + '_' + str(n_epochs) +'.h5')
    with open('models/lm_' + str(train_set) + '_' + str(n_epochs) +'_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    avg_NLL = 0
    avg_NLL_per_word = 0
    # Compute probability of occurence of a sentence
    for sentence in test:
        tok = tokenizer.texts_to_sequences([sentence])[0]
        x_test, y_test = prepare_sentence(tok, maxlen)
        x_test = np.array(x_test)
        y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
        p_pred = model.predict(x_test)
        vocab_inv = {v: k for k, v in vocab.items()}
        log_p_sentence = 0
        for i, prob in enumerate(p_pred):
            word = vocab_inv[y_test[i]+1]  # Index 0 from vocab is reserved to <PAD>
            history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
            prob_word = prob[y_test[i]]
            log_p_sentence -= np.log(prob_word)
            #print('P(w={}|h={})={}'.format(word, history, prob_word))
        avg_NLL += log_p_sentence
        avg_NLL_per_word += log_p_sentence/len(p_pred)
        #print('Prob. sentence: {}'.format(log_p_sentence))
    avg_NLL = avg_NLL/len(train)
    avg_NLL_per_word = avg_NLL_per_word/len(train)

    print('model: ' + str(train_set) + ', avg rNLL: ' + str(avg_NLL) + ', avg rNLL per word: ' + str(avg_NLL_per_word))