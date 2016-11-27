# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from dialog_encdec import *
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
    
    category = cPickle.load(open(state['category'], 'r'))
    assert(len(category)==state['cnum'])

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
    if not state["use_nce"]:
        if ('add_latent_gaussian_per_utterance' in state) and (state["add_latent_gaussian_per_utterance"]):
            logger.debug("Training using variational lower bound on log-likelihood")
        else:
            logger.debug("Training using exact log-likelihood")

        #Test_pred_fn = model.build_train_function() 
        train_batch = model.build_train_function() #训练函数，返回三个量，第一个是training_cost
    else:
        logger.debug("Training with noise contrastive estimation")
        train_batch = model.build_nce_function()

    eval_batch = model.build_eval_function() #测试（验证）函数
    
    #debug_fn = model.build_debug_function()

    if model.add_latent_gaussian_per_utterance:
        eval_grads = model.build_eval_grads()

    #random_sampler = search.RandomSampler(model)
    #beam_sampler = search.BeamSampler(model) 

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
    train_data.start()
    
    
    use_secondary_data = False
    if ('secondary_train_dialogues' in state) and (len(state['secondary_train_dialogues']) > 0):
        logger.debug("Load secondary data")
        use_secondary_data = True
        secondary_train_data = get_secondary_train_iterator(state)
        secondary_train_data.start()
        secondary_rng = numpy.random.RandomState(state['seed'])

    # Build the data structures for Bleu evaluation
    if 'bleu_evaluation' in state:
        bleu_eval_n_1 = BleuEvaluator(n=1)
        bleu_eval_n_2 = BleuEvaluator(n=2)
        bleu_eval_n_3 = BleuEvaluator(n=3)
        bleu_eval_n_4 = BleuEvaluator(n=4)
        jaccard_eval = JaccardEvaluator()
        recall_at_1_eval = RecallEvaluator(n=1)
        recall_at_5_eval = RecallEvaluator(n=5)
        mrr_at_5_eval = MRREvaluator(n=5)
        tfidf_cs_at_1_eval = TFIDF_CS_Evaluator(model, train_data.data_len, 1)
        tfidf_cs_at_5_eval = TFIDF_CS_Evaluator(model, train_data.data_len, 5)

        samples = open(state['bleu_evaluation'], 'r').readlines() 
        n = state['bleu_context_length']
        
        contexts = []
        targets = []
        for x in samples:        
            sentences = x.strip().split('\t')
            assert len(sentences) > n
            contexts.append(sentences[:n])
            targets.append(sentences[n:])

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
    is_end_of_batch = True
    start_validation = False
    training_on_secondary_dataset = False

    batch = None

    while (step < state['loop_iters'] and
            (time.time() - start_time)/60. < state['time_stop'] and
            patience >= 0):

        
        # Training phase

        # If we are training on a primary and secondary dataset, sample at random from either of them
        if is_end_of_batch:
            if use_secondary_data and (secondary_rng.uniform() > state['secondary_proportion']):
                training_on_secondary_dataset = True
            else:
                training_on_secondary_dataset = False

        if training_on_secondary_dataset:
            batch = secondary_train_data.next()
        else:
            batch = train_data.next()
            #print "hehehehehehe!!!!!"

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        
        logger.debug("[TRAIN_%d] - Got batch %d,%d" % (step, batch['x'].shape[1], batch['max_length']))
        
        if batch['max_length'] == state['max_grad_steps']:
            continue
        
        x_data = batch['x']
        
        #print 'x_data:\t',x_data
        
        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        x_semantic = batch['x_semantic']
        x_reset = batch['x_reset']
        ran_cost_utterance = batch['ran_var_constutterance']

        is_end_of_batch = False
        if numpy.sum(numpy.abs(x_reset)) < 1:
            #print 'END-OF-BATCH EXAMPLE!'
            is_end_of_batch = True

        idx_s = (x_data==2).nonzero()[0][0]
        #print x_data
        #print idx_s
        if x_data[1:idx_s].shape[0] < 2:
            continue
        
        """
        if step == 0:
            step += 1
            continue
        out1, out2, out3 = debug_fn(x_data, max_length)
        print x_data
        print
        print out1.shape
        print out1
        print out2.shape
        print out2
        print out3
        """
        #print type(x_data)
        #print x_data
        c, variational_cost, posterior_mean_variance = train_batch(x_data, max_length)


        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            gc.collect()
            continue

        train_cost += c
        train_variational_cost += variational_cost
        train_posterior_mean_variance += posterior_mean_variance

        train_done += batch['num_dialogues']
        train_dialogues_done += batch['num_dialogues']

        this_time = time.time()
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time

            # Keep track of training cost for the last 'train_freq' batches.
            current_train_cost = train_cost/train_done
            if prev_train_done >= 1:
                current_train_cost = float(train_cost - prev_train_cost)/float(train_done - prev_train_done)

            prev_train_cost = train_cost
            prev_train_done = train_done

            h, m, s = ConvertTimedelta(this_time - start_time)
            print ".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f" % (h, m, s,\
                             state['time_stop'] - (time.time() - start_time)/60.,\
                             step, \
                             batch['x'].shape[1], \
                             batch['max_length'], \
                             float(train_cost/train_done))


        if valid_data is not None and\
            step % state['valid_freq'] == 0 and step > 1:
                start_validation = True

        if start_validation and is_end_of_batch:
            start_validation = False
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
            
            fw_valid = open('_VALID__%d.txt'%step, 'w')

            while True:
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
                
                #print ' '.join([idx_to_str[id_of_w] for id_of_w in x_data.T.tolist()[0]])
                idx_s = (x_data==2).nonzero()[0][0]
                if x_data[1:idx_s].shape[0] < 2:
                    continue
                
                c, c_list, variational_cost, posterior_mean_variance, Gen_pro, Tar_Y = eval_batch(x_data, max_length)
                
                if numpy.isinf(c) or numpy.isnan(c):
                    continue
                
                valid_cost += c
                valid_variational_cost += variational_cost
                valid_posterior_mean_variance += posterior_mean_variance

                print 'valid_cost', valid_cost
                #print 'Original: ', ' '.join([idx_to_str[id_of_w] for id_of_w in list(Tar_Y.T)[0]]) #'',join([idx_to_str[id_of_w] for id_of_w in Tar_Y])
                fw_valid.write('Label: '+' '.join([category[id_of_w] for id_of_w in list(Tar_Y.T)[0]])+'\r\n')
                Gen_pro = Gen_pro.tolist()[0]
                enum_ = enumerate(Gen_pro)
                Gen_sort = sorted(enum_, key=lambda x:x[1], reverse=True)[:30]
                Gen_tar = [i[0] for i in Gen_sort]
                
                #print 'Generations: ', ' '.join([idx_to_str[id_of_w] for id_of_w in Gen_tar])
                fw_valid.write('Predict: '+' '.join([category[id_of_w] for id_of_w in Gen_tar])+'\r\n')
                #print 'valid_variational_cost', valid_variational_cost
                #print 'posterior_mean_variance', posterior_mean_variance


                valid_wordpreds_done += batch['num_preds']
                valid_dialogues_done += batch['num_dialogues']

            logger.debug("[VALIDATION END]") 
            fw_valid.close() 
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

        step += 1

    logger.debug("All done, exiting...")

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
