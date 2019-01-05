# Original creator: desire2020@github, https://github.com/desire2020/CoT
# Modified by Jacob Jonsson, cojacobjonsson@github
# Full reference available in Master's thesis

import numpy as np
import tensorflow as tf
from time import time
import random
from dataloader import Gen_Data_loader
from generator import Generator
from target_lstm import TARGET_LSTM
import pickle

from utils.text_process import *

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
#SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 0 # supervise (maximum likelihood estimation) epochs (not recommended)
SEED = 88
BATCH_SIZE = 64
M_DROPOUT_RATE = 1.0 # Dropout rate of M (optional)

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 2000#200000
train_set = 'data/train.txt' # training set in words
test_set = 'data/test.txt'
positive_file = 'save/real_data.txt' # training set in indices
negative_file = 'save/generator_sample.txt' # generated index sequence
generated_num = 10000


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.likelihood_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    SEQ_LENGTH, vocab_size = text_precess(train_set)

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    val_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH) # For testing

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = pickle.load(open('save/target_params_py3.pkl', 'rb'))
    #target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, 32, 32, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    mediator = Generator(vocab_size, BATCH_SIZE*2, EMB_DIM*2, HIDDEN_DIM*2, SEQ_LENGTH, START_TOKEN, name="mediator", dropout_rate=M_DROPOUT_RATE)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # create training set indices
    tokens = get_tokenlized(train_set)
    word_set = get_word_list(tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)
    with open(positive_file, 'w') as outfile:
        outfile.write(text_to_code(tokens, word_index_dict, SEQ_LENGTH))

    # create and load batches from index training set
    gen_data_loader.create_batches(positive_file)


    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    #generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    #gen_data_loader.create_batches(positive_file) # use training file
    #generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, eval_file)
    #val_data_loader.create_batches(eval_file)

    log = open('save/experiment-log' + str(time()) + '.txt', 'w')
    log_nll = open('save/experiment-log-nll' + str(time()) + '.txt', 'w')

    print('#########################################################################')
    print('Start Cooperative Training...')
    print('Num batches: ' + str(gen_data_loader.num_batch))
    for iter_idx in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            print('Training G')
            samples = generator.generate(sess)
            rewards = mediator.get_reward(sess, np.concatenate([samples, samples], axis=0))
            feed = {generator.x: samples, generator.rewards: rewards[0:BATCH_SIZE]}
            loss, _ = sess.run([generator.g_loss, generator.g_updates], feed_dict=feed)
            #print(loss) # remove, to often?
            #_ = sess.run(generator.g_updates, feed_dict=feed)
            if iter_idx % gen_data_loader.num_batch == 0:
                print('cooptrain epoch#', iter_idx // gen_data_loader.num_batch)
                print('loss: ' + str(loss))
        # Test, removed oracle
        if iter_idx % 100 == 0 or iter_idx == TOTAL_BATCH - 1:
            print('Generating fake samples')
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            likelihood_data_loader.create_batches(negative_file)
            print('Calculating NLL')
            test_loss = target_loss(sess, generator, gen_data_loader) # use validation generator? Texygen uses same
            print('batch:\t', iter_idx, 'nll_test ', test_loss)
            buffer = 'batch:\t'+ str(iter_idx) + '\tnll_test:\t' + str(test_loss) + '\n'
            log_nll.write(buffer)
        # Train the mediator
        for _ in range(1):
            print('Training M')
            bnll_ = []
            collected_x = []
            ratio = 2
            for it in range(ratio):
                if it % 2 == 0:
                    x_batch = gen_data_loader.next_batch()
                else:
                    x_batch = generator.generate(sess)
                collected_x.append(x_batch)
            collected_x = np.reshape(collected_x, [-1, SEQ_LENGTH])
            np.random.shuffle(collected_x)
            collected_x = np.reshape(collected_x, [-1, BATCH_SIZE*2, SEQ_LENGTH])
            for it in range(1):
                print('Calculating BNLL')
                feed = {
                    mediator.x: collected_x[it],
                }
                bnll = sess.run(mediator.likelihood_loss, feed)
                bnll_.append(bnll)
                # sess.run(mediator.dropout_on)
                _ = sess.run(mediator.likelihood_updates, feed)
                # sess.run(mediator.dropout_off)
        if (iter_idx * 4) % gen_data_loader.num_batch == 0:
            print('Calculating likelihood loss for M')
            bnll = np.mean(bnll_)
            gnll = sess.run(mediator.likelihood_loss, feed_dict={mediator.x: np.reshape([generator.generate(sess), generator.generate(sess)], [BATCH_SIZE*2, SEQ_LENGTH])})
            print("mediator cooptrain iter#%d, balanced_nll %f, g_nll %f" % (iter_idx, bnll, gnll))
            log.write("%d\t%f\n" % (iter_idx, bnll))

    log.close()
    log_nll.close()

if __name__ == '__main__':
    main()