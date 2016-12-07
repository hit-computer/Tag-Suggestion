# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from doc_encdec import *
from utils import *
from evaluation import *

import time
import traceback
import os.path
import sys
import argparse
import cPickle
import logging
import search
import pprint
import numpy
import collections
import signal
import math
import gc
from sklearn.metrics.pairwise import cosine_similarity

#import matplotlib
#matplotlib.use('Agg')
#import pylab
#theano.config.compute_test_value = 'warn'

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Additional measures can be set here
measures = ["train_cost", "train_misclass", "train_variational_cost", "train_posterior_mean_variance", "valid_cost", "valid_misclass", "valid_posterior_mean_variance", "valid_variational_cost", "valid_emi", "valid_bleu_n_1", "valid_bleu_n_2", "valid_bleu_n_3", "valid_bleu_n_4", 'valid_jaccard', 'valid_recall_at_1', 'valid_recall_at_5', 'valid_mrr_at_5', 'tfidf_cs_at_1', 'tfidf_cs_at_5']


def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings, post_fix = ''):
    print "Saving the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'timing.npz', **timings)
    signal.signal(signal.SIGINT, s)
    
    print "Model saved, took {}".format(time.time() - start)

def load(model, filename, parameter_strings_to_ignore):
    print "Loading the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename, parameter_strings_to_ignore)
    signal.signal(signal.SIGINT, s)

    print "Model loaded, took {}".format(time.time() - start)

def main(args):     
    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
     
    state = eval(args.prototype)() 
    timings = init_timings() 
    
    args.resume = 'Que26/models/1448530885.38_testmodel__225000'
    
    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timing.npz'
        
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            for x, y in timings.items():
                timings[x] = list(y)

            # Increment seed to make sure we get newly shuffled batches when training on large datasets
            state['seed'] = state['seed'] + 10

        else:
            raise Exception("Cannot resume, cannot find files!")

    #logger.debug("State:\n{}".format(pprint.pformat(state)))
    #logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
 
    if args.force_train_all_wordemb == True:
        state['fix_pretrained_word_embeddings'] = False
        
    # Load dictionary
    raw_dict = cPickle.load(open(state['dictionary'], 'r'))
    # Dictionaries to convert str to idx and vice-versa
    str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict]) #字典里的每一项包含四个字段，（字符，字符号，词频，文本频率）
    idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])

    model = DialogEncoderDecoder(state)
    rng = model.rng 
    
    
    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")

            parameter_strings_to_ignore = []
            if args.reinitialize_decoder_parameters:
                parameter_strings_to_ignore += ['latent_utterance_prior']
                parameter_strings_to_ignore += ['latent_utterance_approx_posterior']
            if args.reinitialize_variational_parameters:
                parameter_strings_to_ignore += ['Wd_']
                parameter_strings_to_ignore += ['bd_']

            load(model, filename, parameter_strings_to_ignore)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            raise Exception('Backward compatibility not ensured! (need run_id in state)')           

    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID
    
    
    logger.debug("Compile trainer")

    test_batch = model.build_test_function() #测试函数

    if model.add_latent_gaussian_per_utterance:
        eval_grads = model.build_eval_grads()

    random_sampler = search.RandomSampler(model)
    beam_sampler = search.BeamSampler(model) 

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
    
    

    # Start looping through the dataset
    step = 0
    patience = state['patience'] 
    start_time = time.time()
     
    train_cost = 0
    train_variational_cost = 0
    train_posterior_mean_variance = 0
    train_misclass = 0
    train_done = 0
    train_dialogues_done = 0.0

    prev_train_cost = 0
    prev_train_done = 0

    ex_done = 0

    batch = None

    valid_data.start()
    valid_cost = 0
    valid_variational_cost = 0
    valid_posterior_mean_variance = 0

    valid_wordpreds_done = 0
    valid_dialogues_done = 0


    # Prepare variables for plotting histogram over word-perplexities and mutual information
    valid_data_len = valid_data.data_len
    valid_cost_list = numpy.zeros((valid_data_len,))
    valid_pmi_list = numpy.zeros((valid_data_len,))

    # Prepare variables for printing the training examples the model performs best and worst on
    valid_extrema_setsize = min(state['track_extrema_samples_count'], valid_data_len)
    valid_extrema_samples_to_print = min(state['print_extrema_samples_count'], valid_extrema_setsize)

    max_stored_len = 160 # Maximum number of tokens to store for dialogues with highest and lowest validation errors
    valid_lowest_costs = numpy.ones((valid_extrema_setsize,))*1000
    valid_lowest_dialogues = numpy.ones((valid_extrema_setsize,max_stored_len))*1000
    valid_highest_costs = numpy.ones((valid_extrema_setsize,))*(-1000)
    valid_highest_dialogues = numpy.ones((valid_extrema_setsize,max_stored_len))*(-1000)

    logger.debug("[VALIDATION START]")
    DocMtrix = []
    NNN = 0
    while True:
        NNN += 1
        if NNN > 50:
            break
        batch = valid_data.next()

        # Train finished
        if not batch:
            break
         
        logger.debug("[VALID] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
        if batch['max_length'] == state['max_grad_steps']:
            continue

        x_data = batch['x']
        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        x_semantic = batch['x_semantic']
        x_semantic_nonempty_indices = numpy.where(x_semantic >= 0)

        x_reset = batch['x_reset']
        ran_cost_utterance = batch['ran_var_constutterance']

        Gen_tar, Tar_Y, DocV= test_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_semantic, x_reset, ran_cost_utterance)
        
        DocMtrix.append(DocV)
        print ''.join([idx_to_str[id_of_w] for id_of_w in list(x_data.T)[0]])
        # Rehape into matrix, where rows are validation samples and columns are tokens
        # Note that we use max_length-1 because we don't get a cost for the first token
        # (the first token is always assumed to be eos)
        #c_list = c_list.reshape((batch['x'].shape[1],max_length-1), order=(1,0))
        #c_list = numpy.sum(c_list, axis=1)
        
        #words_in_dialogues = numpy.sum(x_cost_mask, axis=0)
        #c_list = c_list / words_in_dialogues
        #print 'Original: ', ''.join([idx_to_str[id_of_w] for id_of_w in list(Tar_Y.T)[0]]) #'',join([idx_to_str[id_of_w] for id_of_w in Tar_Y])
        #print 'Generations: ',''.join([idx_to_str[id_of_w] for id_of_w in list(Gen_tar.T)[0]])
        #print 'Test:', type(test1), test1
        #raw_input()
        """
        if numpy.isinf(c) or numpy.isnan(c):
            continue
        
        valid_cost += c
        valid_variational_cost += variational_cost
        valid_posterior_mean_variance += posterior_mean_variance

        print 'valid_cost', valid_cost
        print 'Original: ', ''.join([idx_to_str[id_of_w] for id_of_w in list(Tar_Y.T)[0]]) #'',join([idx_to_str[id_of_w] for id_of_w in Tar_Y])
        print 'Generations: ', ''.join([idx_to_str[id_of_w] for id_of_w in list(Gen_tar.T)[0]])
        #print 'valid_variational_cost', valid_variational_cost
        #print 'posterior_mean_variance', posterior_mean_variance


        valid_wordpreds_done += batch['num_preds']
        valid_dialogues_done += batch['num_dialogues']
        """
    logger.debug("[VALIDATION END]") 
    DocM = numpy.row_stack(DocMtrix)
    simM = cosine_similarity(DocM,DocM)
    meanV = numpy.mean(DocM,axis=1)
    print simM
    print meanV
    """
    valid_cost /= valid_wordpreds_done
    valid_variational_cost /= valid_wordpreds_done
    valid_posterior_mean_variance /= valid_dialogues_done

    if len(timings["valid_cost"]) == 0 or valid_cost < numpy.min(timings["valid_cost"]):
        patience = state['patience']
        # Saving model if decrease in validation cost
        save(model, timings)
        print 'best valid_cost', valid_cost
    elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
        patience -= 1

    if args.save_every_valid_iteration:
        save(model, timings, '_' + str(step) + '_')



    print "** valid cost (NLL) = %.4f, valid word-perplexity = %.4f, valid variational cost (per word) = %.8f, valid mean posterior variance (per word) = %.8f, patience = %d" % (float(valid_cost), float(math.exp(valid_cost)), float(valid_variational_cost), float(valid_posterior_mean_variance), patience)

    timings["train_cost"].append(train_cost/train_done)
    timings["train_variational_cost"].append(train_variational_cost/train_done)
    timings["train_posterior_mean_variance"].append(train_posterior_mean_variance/train_dialogues_done)
    timings["valid_cost"].append(valid_cost)
    timings["valid_variational_cost"].append(valid_variational_cost)
    timings["valid_posterior_mean_variance"].append(valid_posterior_mean_variance)

    # Reset train cost, train misclass and train done
    train_cost = 0
    train_done = 0
    prev_train_cost = 0
    prev_train_done = 0

    logger.debug("All done, exiting...")
    """
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--force_train_all_wordemb", action='store_true', help="If true, will force the model to train all word embeddings in the encoder. This switch can be used to fine-tune a model which was trained with fixed (pretrained)  encoder word embeddings.")
    parser.add_argument("--save_every_valid_iteration", action='store_true', help="If true, will save a copy of the model at every validation iteration.")

    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    parser.add_argument("--reinitialize-variational-parameters", action='store_true', help="Can be used when resuming a model. If true, will initialize all variational parameters randomly instead of loading them from previous model.")

    parser.add_argument("--reinitialize-decoder-parameters", action='store_true', help="Can be used when resuming a model. If true, will initialize all parameters of the utterance decoder randomly instead of loading them from previous model.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    main(args)
