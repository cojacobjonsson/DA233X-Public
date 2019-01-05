import numpy as np
import tensorflow as tf
import random
import os
import time
from dataloader import Gen_Data_loader
from generator import Generator
from target_lstm import TARGET_LSTM
import pickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 0 # supervise (maximum likelihood estimation) epochs (not recommended). takes about 33 min per epoch
SEED = 88
BATCH_SIZE = 64
M_DROPOUT_RATE = 0.5 # Dropout rate of M (optional)

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 20
generated_num = 1000#10000

task_id = os.getenv('SLURM_JOB_ID') 

true_file = 'data/train.txt' # the raw training file
val_file = 'data/test.txt' # the raw validation file
oracle_file = 'save/oracle_' + str(task_id) + '.txt' # the encoded file to actually train on, ground truth
val_oracle_file = 'save/oracle_val_' + str(task_id) + '.txt' # the encoded file to actually train on, ground truth
generator_file = 'save/generator_' + str(task_id) + '.txt' # the generated encoded file
test_file = 'save/test_file_' + str(task_id) + '.txt' # the decoded file to read and evaluate


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

def get_real_test_file(dict, generator_file, test_file):
    from utils.text_process import code_to_text
    from utils.text_process import get_tokenlized
    with open(generator_file, 'r') as file:
        codes = get_tokenlized(generator_file)
    with open(test_file, 'w') as outfile:
        outfile.write(code_to_text(codes=codes, dictionary=dict))

def mle_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()
    
    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.maximum_likelihood(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def main():
    print('program start')
    from utils.text_process import text_precess, text_to_code # TODO: move?
    from utils.text_process import get_tokenlized, get_word_list, get_dict
        
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    SEQ_LENGTH, vocab_size = text_precess(true_file, val_file)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    val_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    
    # Create training file and dicts
    tokens = get_tokenlized(true_file)
    val_tokens = get_tokenlized(val_file)
    word_set = get_word_list(tokens + val_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)
    with open(oracle_file, 'w') as outfile:
        outfile.write(text_to_code(tokens, word_index_dict, SEQ_LENGTH))
    with open(val_oracle_file, 'w') as outfile:
        outfile.write(text_to_code(val_tokens, word_index_dict, SEQ_LENGTH))

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    #target_params = pickle.load(open('save/target_params_py3.pkl', 'rb'))
    #target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, 32, 32, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model
    # replace target lstm with true data
    
    mediator = Generator(vocab_size, BATCH_SIZE*2, EMB_DIM*2, HIDDEN_DIM*2, SEQ_LENGTH, START_TOKEN, name="mediator", dropout_rate=M_DROPOUT_RATE)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    gen_data_loader.create_batches(oracle_file)
    val_data_loader.create_batches(val_oracle_file)

    log = open('save/experiment-log.txt', 'w')
    log_nll = open('save/experiment-log-nll.txt', 'w')

    #  pre-train generator (default 0 epochs)(not recommended)
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = mle_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, generator_file)
            #get_real_test_file(index_word_dict, generator_file, test_file) # only needed in debugging
            test_loss = target_loss(sess, generator, val_data_loader)
            print('pre-train epoch ', epoch, 'nll_test ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll_test:\t' + str(test_loss) + '\n'
            log_nll.write(buffer)

    print('#########################################################################')
    toc = time.time()
    print('Start Cooperative Training...')
    for iter_idx in range(TOTAL_BATCH):
        print('iteration: ' + str(iter_idx) + '\ntime: ' + str(time.time() - toc))
        toc = time.time()
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = mediator.get_reward(sess, np.concatenate([samples, samples], axis=0))
            feed = {generator.x: samples, generator.rewards: rewards[0:BATCH_SIZE]}
            loss, _ = sess.run([generator.g_loss, generator.g_updates], feed_dict=feed)
        # Test, removed oracle test
        if iter_idx % gen_data_loader.num_batch == 0: # epochs instead of batches
            test_loss = target_loss(sess, generator, val_data_loader)
            print('epoch:\t', iter_idx // gen_data_loader.num_batch, 'nll_test ', test_loss)
            buffer = 'epoch:\t'+ str(iter_idx // gen_data_loader.num_batch) + '\tnll_test:\t' + str(test_loss) + '\n'
            log_nll.write(buffer)
        if iter_idx == TOTAL_BATCH - 1:
            print('generating samples')
            generate_samples(sess, generator, BATCH_SIZE, generated_num, generator_file)
            get_real_test_file(index_word_dict, generator_file, test_file)
        # Train the mediator
        for _ in range(1):
            print('training mediator...')
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
                feed = {
                    mediator.x: collected_x[it],
                }
                print('running bnll sess')
                bnll = sess.run(mediator.likelihood_loss, feed)
                bnll_.append(bnll)
                print('running mediator and updating')
                sess.run(mediator.dropout_on)
                _ = sess.run(mediator.likelihood_updates, feed)
                sess.run(mediator.dropout_off)
            if iter_idx % 50 == 0:
                bnll = np.mean(bnll_)
                print("mediator cooptrain iter#%d, balanced_nll %f" % (iter_idx, bnll))
                log.write("%d\t%f\n" % (iter_idx, bnll))
        
    log.close()
    log_nll.close()

if __name__ == '__main__':
    main()
