#-*- coding:utf-8 -*-
"""
Dialog hierarchical encoder-decoder code.
The code is inspired from nmt encdec code in groundhog
but we do not rely on groundhog infrastructure.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni, Iulian Vlad Serban")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
logger = logging.getLogger(__name__)

from theano.sandbox.scan import scan
#from theano import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv3d2d import *

from collections import OrderedDict

from model import *
from utils import *

import operator

# Theano speed-up
#theano.config.scan.allow_gc = False
#

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

class EncoderDecoderBase():
    def __init__(self, state, rng, parent):
        self.rng = rng
        self.parent = parent
        
        self.state = state
        self.__dict__.update(state)
        
        self.dialogue_rec_activation = eval(self.dialogue_rec_activation)
        self.sent_rec_activation = eval(self.sent_rec_activation)
         
        self.params = []

class UtteranceEncoder(EncoderDecoderBase):
    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.Filter1 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='Filter1'))
        self.Filter2 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, 2*self.rankdim, self.qdim_encoder), name='Filter2'))
        self.Filter3 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, 3*self.rankdim, self.qdim_encoder), name='Filter3'))
        
        self.b_1 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='cnn_b1'))
        self.b_2 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='cnn_b2'))
        self.b_3 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='cnn_b3'))

    # This function takes as input word indices and extracts their corresponding word embeddings
    def approx_embedder(self, x):
        return self.W_emb[x]
    
    def ConvLayer1(self, q1):
        output = T.dot(q1, self.Filter1) + self.b_1
        return output
    
    def ConvLayer2(self, q1, q2):
        output = T.dot(T.concatenate([q1, q2], axis=1), self.Filter2) + self.b_2
        return output
    
    def ConvLayer3(self, q1, q2, q3):
        output = T.dot(T.concatenate([q1, q2, q3], axis=1), self.Filter3) + self.b_3
        return output
    
    def Convolution(self, x):
        xe = self.approx_embedder(x)
        
        _res1, _ = theano.scan(self.ConvLayer1, sequences=[xe])
        _res2, _ = theano.scan(self.ConvLayer2, sequences=[xe[:-1], xe[1:]])
        _res3, _ = theano.scan(self.ConvLayer3, sequences=[xe[:-2],xe[1:-1],xe[2:]])
        
        hidden1 = T.tanh(T.max(_res1,axis=0))
        hidden2 = T.tanh(T.max(_res2,axis=0))
        hidden3 = T.tanh(T.max(_res3,axis=0))
        
        return T.mean(T.concatenate([hidden1, hidden2, hidden3], axis=0), axis=0)
        #return (hidden1 + hidden2 + hidden3)/3.0
        #return x[:5]
        #return (hidden1 + hidden2)/2.0
    
    def _split(self, e, s, x):
        return e+1, self.Convolution(x[s:e])
    
    def build_encoder(self, x, **kwargs): #x是一个matrix
        idxs = T.eq(x, self.eos_sym).nonzero()[0]
        
        res, _ = theano.scan(self._split, sequences = [idxs], outputs_info=[numpy.int64(0), None], non_sequences=[x])
        res = res[1]
        
        return res

    def __init__(self, state, rng, word_embedding_param, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(word_embedding_param)

class DialogEncoder(EncoderDecoderBase):
    def init_params(self):
        """ Context weights """

        if self.bidirectional_utterance_encoder:
            # With the bidirectional flag, the dialog encoder gets input 
            # from both the forward and backward utterance encoders, hence it is double qdim_encoder
            input_dim = self.qdim_encoder
        else:
            # Without the bidirectional flag, the dialog encoder only gets input
            # from the forward utterance encoder, which has dim self.qdim_encoder
            input_dim = self.qdim_encoder

        transformed_input_dim = input_dim
        if self.deep_dialogue_input:
            self.Ws_deep_input = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, input_dim, self.sdim), name='Ws_deep_input'+self.name))
            self.bs_deep_input = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_deep_input'+self.name))
            transformed_input_dim = self.sdim

        
        self.Ws_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in'+self.name))
        self.Ws_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh'+self.name))
        self.bs_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh'+self.name))


        if self.dialogue_encoder_gating == "GRU":
            self.Ws_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_r'+self.name))
            self.Ws_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_z'+self.name))
            self.Ws_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_r'+self.name))
            self.Ws_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_z'+self.name))
            self.bs_z = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_z'+self.name))
            self.bs_r = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_r'+self.name))
    
    def plain_dialogue_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        # If deep input to dialogue encoder is enabled, run h_t through an MLP
        transformed_h_t = h_t
        if self.deep_dialogue_input: #prototype_test True
            transformed_h_t = self.dialogue_rec_activation(T.dot(h_t, self.Ws_deep_input) + self.bs_deep_input)

        hs_tilde = self.dialogue_rec_activation(T.dot(transformed_h_t, self.Ws_in) + T.dot(hs_tm1, self.Ws_hh) + self.bs_hh)

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_tilde 
        return hs_t

    def GRU_dialogue_step(self, h_t, hs_tm1):
        # If deep input to dialogue encoder is enabled, run h_t through an MLP
        transformed_h_t = h_t
        if self.deep_dialogue_input:
            transformed_h_t = self.dialogue_rec_activation(T.dot(h_t, self.Ws_deep_input) + self.bs_deep_input)

        rs_t = T.nnet.sigmoid(T.dot(transformed_h_t, self.Ws_in_r) + T.dot(hs_tm1, self.Ws_hh_r) + self.bs_r)
        zs_t = T.nnet.sigmoid(T.dot(transformed_h_t, self.Ws_in_z) + T.dot(hs_tm1, self.Ws_hh_z) + self.bs_z)
        hs_tilde = self.dialogue_rec_activation(T.dot(transformed_h_t, self.Ws_in) + T.dot(rs_t * hs_tm1, self.Ws_hh) + self.bs_hh)
        hs_update = (np.float32(1.) - zs_t) * hs_tm1 + zs_t * hs_tilde
        
        hs_t = hs_update #从这里可以很明显的看出若词i不是</s>，那么的它的m_t为1，输出就是hs_tml（保持不变）；但处理到一个语句末尾是，即</s>时，输出值为hs_update.
        return hs_t, hs_tilde, rs_t, zs_t

    def build_encoder(self, h, x, xmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            if prev_state:
                hs_0 = prev_state
            else:
                hs_0 = T.alloc(np.float32(0), batch_size, self.sdim)

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)       

        if self.dialogue_encoder_gating == "GRU": #True
            f_hier = self.GRU_dialogue_step
            o_hier_info = [hs_0, None, None, None]
        else:
            f_hier = self.plain_dialogue_step
            o_hier_info = [hs_0]
        
        # All hierarchical sentence
        # The hs sequence is based on the original mask
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[h],\
                               outputs_info=o_hier_info)#GRU中不存在神经元的状态，直接将上一个输出作为下一次循环迭代的输入。
        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        return hs 

    def __init__(self, state, rng, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params()

class DialogEncoderDecoder(Model):

    def compute_updates(self, training_cost, params):
        updates = []
         
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))

        # Clip stuff
        c = numpy.float32(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))
        
        grads = OrderedDict(clip_grads)

        if self.initialize_from_pretrained_word_embeddings and self.fix_pretrained_word_embeddings:
            # Keep pretrained word embeddings fixed
            logger.debug("Will use mask to fix pretrained word embeddings")
            grads[self.W_emb] = grads[self.W_emb] * self.W_emb_pretrained_mask
        else:
            logger.debug("Will train all word embeddings")

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)  
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads)
        else:
            raise Exception("Updater not understood!") 

        return updates

    def build_debug_function(self):
        if not hasattr(self, 'test_fn'):
            # Compile functions
            logger.debug("Building test function")
            
            self.debug_fn = theano.function(inputs=[self.x_data, self.x_max_length],
                                            outputs=[self.h, self.Doc_Vec, self.training_cost],
                                            updates=self.updates + self.state_updates,
                                            on_unused_input='ignore', 
                                            name="debug_fn")
            
        return self.debug_fn    
    
        
    def build_test_function(self):
        if not hasattr(self, 'test_fn'):
            # Compile functions
            logger.debug("Building test function")
            
            self.test_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                         self.x_max_length,
                                                         self.x_semantic_targets, self.x_reset_mask, 
                                                         self.ran_cost_utterance],
                                            outputs=[self.Generate_Y, self.Target_Y, self.Doc_Vec],
                                            on_unused_input='ignore', 
                                            name="test_fn")
            
        return self.test_fn
        
    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function")
            
            self.train_fn = theano.function(inputs=[self.x_data, 
                                                         self.x_max_length],
                                            outputs=[self.training_cost, self.variational_cost, self.latent_utterance_variable_approx_posterior_mean_var],
                                            updates=self.updates + self.state_updates, 
                                            on_unused_input='ignore', 
                                            name="train_fn")
            
        return self.train_fn
    
    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")
            self.eval_fn = theano.function(inputs=[self.x_data, self.x_max_length], 
                                            outputs=[self.evaluation_cost, self.sigmoid_cost, self.variational_cost, self.latent_utterance_variable_approx_posterior_mean_var,self.Gen_pro, self.Target_Y], 
                                            updates=self.state_updates,
                                            on_unused_input='ignore', name="eval_fn")
        return self.eval_fn
   
        
    def __init__(self, state):
        Model.__init__(self)

        # Compatibility towards older models
        if 'bootstrap_from_semantic_information' in state:
            assert state['bootstrap_from_semantic_information'] == False # We don't support semantic info right now...


        if not 'bidirectional_utterance_encoder' in state:
            state['bidirectional_utterance_encoder'] = False

        if 'encode_with_l2_pooling' in state:
            assert state['encode_with_l2_pooling'] == False # We don't support L2 pooling right now...

        if not 'direct_connection_between_encoders_and_decoder' in state:
            state['direct_connection_between_encoders_and_decoder'] = False

        if not 'deep_direct_connection' in state:
            state['deep_direct_connection'] = False

        if not state['direct_connection_between_encoders_and_decoder']:
            assert(state['deep_direct_connection'] == False)

        if not 'collaps_to_standard_rnn' in state:
            state['collaps_to_standard_rnn'] = False

        #if not 'never_reset_decoder' in state:
        #    state['never_reset_decoder'] = False

        if not 'reset_utterance_decoder_at_end_of_utterance' in state:
            state['reset_utterance_decoder_at_end_of_utterance'] = True

        if not 'reset_utterance_encoder_at_end_of_utterance' in state:
            state['reset_utterance_encoder_at_end_of_utterance'] = True

        if not 'deep_dialogue_input' in state:
            state['deep_dialogue_input'] = True

        if not 'reset_hidden_states_between_subsequences' in state:
            state['reset_hidden_states_between_subsequences'] = False

        if not 'add_latent_gaussian_per_utterance' in state:
           state['add_latent_gaussian_per_utterance'] = False
        if not 'latent_gaussian_per_utterance_dim' in state:
           state['latent_gaussian_per_utterance_dim'] = 1
        if not 'condition_latent_variable_on_dialogue_encoder' in state:
           state['condition_latent_variable_on_dialogue_encoder'] = True
        if not 'condition_latent_variable_on_dcgm_encoder' in state:
           state['condition_latent_variable_on_dcgm_encoder'] = False
        if not 'scale_latent_variable_variances' in state:
           state['scale_latent_variable_variances'] = 0.01
        if not 'condition_decoder_only_on_latent_variable' in state:
           state['condition_decoder_only_on_latent_variable'] = False
        if not 'train_latent_gaussians_with_batch_normalization' in state:
           state['train_latent_gaussians_with_batch_normalization'] = False

        if state['condition_latent_variable_on_dialogue_encoder']:
            assert state['add_latent_gaussian_per_utterance']

        if state['condition_latent_variable_on_dcgm_encoder']:
            assert state['add_latent_gaussian_per_utterance']

        self.state = state
        self.global_params = []

        self.__dict__.update(state) #相当于执行 for key in state: self.key = state['key'], 这里把key看成一个符号吧，不是变量，也就是self.name = state['name']等等
        self.rng = numpy.random.RandomState(state['seed']) 

        # Load dictionary
        raw_dict = cPickle.load(open(self.dictionary, 'r'))
        # Probabilities for each term in the corpus
        self.noise_probs = [x[2] for x in sorted(raw_dict, key=operator.itemgetter(1))]
        self.noise_probs = numpy.array(self.noise_probs, dtype='float64')
        self.noise_probs /= numpy.sum(self.noise_probs)
        self.noise_probs = self.noise_probs ** 0.75
        self.noise_probs /= numpy.sum(self.noise_probs)
        
        self.t_noise_probs = theano.shared(self.noise_probs.astype('float32'), 't_noise_probs')
        # Dictionaries to convert str to idx and vice-versa
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict]) #字典里的每一项包含四个字段，（字符，字符号，词频，文本频率）
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])

        # Extract document (dialogue) frequency for each word
        self.word_freq = dict([(tok_id, freq) for _, tok_id, freq, _ in raw_dict])
        self.document_freq = dict([(tok_id, df) for _, tok_id, _, df in raw_dict])

        #if '</s>' not in self.str_to_idx \
        #   or '</d>' not in self.str_to_idx:
        #   raise Exception("Error, malformed dictionary!")

        if '</s>' not in self.str_to_idx:
           raise Exception("Error, malformed dictionary!")
         
        # Number of words in the dictionary 
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim
        logger.debug("idim: " + str(self.idim))

        logger.debug("Initializing Theano variables")
        self.y_neg = T.itensor3('y_neg')
        self.x_data = T.imatrix('x_data')
        self.x_data_reversed = T.imatrix('x_data_reversed')
        self.x_cost_mask = T.matrix('cost_mask')
        self.x_reset_mask = T.vector('reset_mask')
        self.x_max_length = T.iscalar('x_max_length')
        self.x_semantic_targets = T.imatrix('x_semantic')
        self.ran_cost_utterance = T.tensor3('ran_cost_utterance')


        
        # The training data is defined as all symbols except the last, and
        # the target data is defined as all symbols except the first.
        training_x = self.x_data[:(self.x_max_length-1)]
        
        #修改training_x和training_y，使其只表示target。只取</d>后面的内容
        idx_s = T.eq(training_x,self.eod_sym).nonzero()[0][0]
        Dtraining_y = training_x[idx_s+1:]
        training_x = training_x[1:idx_s]
        #idx_e = training_x.nonzero()[0][-1]
        #Dtraining_x = training_x[idx_s:]
        
        # Here we find the end-of-sentence tokens in the minibatch.
        training_hs_mask = T.neq(training_x, self.eos_sym) #所有非结束符</s>为True，即training_x不等于1的位置为True，维度和training_x一致
        #training_x_cost_mask = self.x_cost_mask[1:self.x_max_length].flatten() #x表示一段对话，x_max_length表示这段对话包含的字符数目
        
        # Build word embeddings, which are shared throughout the model
        if self.initialize_from_pretrained_word_embeddings == True:
            # Load pretrained word embeddings from pickled file
            logger.debug("Loading pretrained word embeddings")
            pretrained_embeddings = cPickle.load(open(self.pretrained_word_embeddings_file, 'r')) #pretrained_embeddings为一个list，包含两个元素，第一个是word embedding（array类型）

            # Check all dimensions match from the pretrained embeddings
            assert(self.idim == pretrained_embeddings[0].shape[0])
            assert(self.rankdim == pretrained_embeddings[0].shape[1])
            assert(self.idim == pretrained_embeddings[1].shape[0])
            assert(self.rankdim == pretrained_embeddings[1].shape[1])

            self.W_emb_pretrained_mask = theano.shared(pretrained_embeddings[1].astype(numpy.float32), name='W_emb_mask')
            self.W_emb = add_to_params(self.global_params, theano.shared(value=pretrained_embeddings[0].astype(numpy.float32), name='W_emb'))#是一个matrix，每一行是一个词的embedding
        else:
            # Initialize word embeddings randomly
            self.W_emb = add_to_params(self.global_params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))

        # Variables to store encoder and decoder states
        self.ph = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='ph')
        self.phs = theano.shared(value=numpy.zeros((self.bs, self.sdim), dtype='float32'), name='phs')

        logger.debug("Initializing utterance encoder")
        self.utterance_encoder = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')
        logger.debug("Build utterance encoder")
        # The encoder h embedding is the final hidden state of the forward encoder RNN
        self.h = self.utterance_encoder.build_encoder(training_x,)

        
        logger.debug("Initializing dialog encoder")
        self.dialog_encoder = DialogEncoder(self.state, self.rng, self, '')

        logger.debug("Build dialog encoder")
        self.hs = self.dialog_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask, prev_state=self.phs)

        # We initialize the stochastic "latent" variables
        # platent_utterance_variable_prior

        self.variational_cost = theano.shared(value=numpy.float(0))
        self.latent_utterance_variable_approx_posterior_mean_var = theano.shared(value=numpy.float(0))

        #self.hd_input = self.hs
        
        self.Doc_Vec = self.hs[-1]
        
        self.Wd_out = add_to_params(self.global_params, theano.shared(value=NormalInit(self.rng, self.sdim, self.cnum), name='Wd_out'))
        self.bd_out = add_to_params(self.global_params, theano.shared(value=np.zeros((self.cnum,), dtype='float32'), name='bd_out'))
        pre_activ = T.dot(self.Doc_Vec, self.Wd_out)+ self.bd_out
        outputs = T.nnet.sigmoid(pre_activ)
        y = Dtraining_y.flatten()
        #outputs = output.flatten()
        y_mask = theano.shared(value=np.zeros((1,self.cnum), dtype='float32'), name='y_mask')
        y_mask = T.set_subtensor(y_mask[0,y], 1)
        target_probs = y_mask*outputs + (1-y_mask)*(1-outputs)
        
        self.Target_Y = Dtraining_y
        self.Gen_pro = outputs
        #self.Test_prevs = self.hd_input
        #self.test1 = outputs
        #self.test2 = y
        #self.test3 = target_probs
            
        
        # Prediction cost and rank cost
        #self.contrastive_cost = T.sum(contrastive_cost.flatten() * training_x_cost_mask)
        self.sigmoid_cost = -T.log(target_probs) #* training_x_cost_mask
        #self.softmax_cost_acc = 

        # Compute training cost, which equals standard cross-entropy error
        self.training_cost = T.sum(self.sigmoid_cost)
        
        self.evaluation_cost = self.training_cost
        
        
        # Init params
        if self.collaps_to_standard_rnn:
                self.params = self.global_params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_decoder.params))
        else:
            if self.bidirectional_utterance_encoder:
                self.params = self.global_params + self.utterance_encoder_forward.params + self.utterance_encoder_backward.params + self.dialog_encoder.params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_encoder_forward.params) + len(self.utterance_encoder_backward.params) + len(self.dialog_encoder.params) + len(self.utterance_decoder.params))
            else:
                self.params = self.global_params + self.utterance_encoder.params + self.dialog_encoder.params #+ self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_encoder.params) + len(self.dialog_encoder.params)) #+ len(self.utterance_decoder.params))

        if self.add_latent_gaussian_per_utterance:
            self.params += self.latent_utterance_variable_prior_encoder.params
            self.params += self.latent_utterance_variable_approx_posterior_encoder.params

            if self.condition_latent_variable_on_dcgm_encoder:
                self.params += self.dcgm_encoder.params

        self.updates = self.compute_updates(self.training_cost / training_x.shape[1], self.params)

        # Truncate gradients properly by bringing forward previous states
        # First, create reset mask
        #x_reset = self.x_reset_mask.dimshuffle(0, 'x')
        # if flag 'reset_hidden_states_between_subsequences' is on, then
        # always reset
        if self.reset_hidden_states_between_subsequences:
            x_reset = 0

        # Next, compute updates using reset mask (this depends on the number of RNNs in the model)
        x_reset = 0
        self.state_updates = []
        if self.bidirectional_utterance_encoder:
            #self.state_updates.append((self.ph_fwd, x_reset * res_forward[-1]))
            #self.state_updates.append((self.ph_bck, x_reset * res_backward[-1]))
            self.state_updates.append((self.phs, x_reset * self.hs[-1]))
        else:
            #self.state_updates.append((self.ph, x_reset * self.h[-1]))
            self.state_updates.append((self.phs, x_reset * self.hs[-1]))
            #self.state_updates.append((self.phd, x_reset * self.hd[-1]))

        if self.direct_connection_between_encoders_and_decoder:
            self.state_updates.append((self.phs_dummy, x_reset * self.hs_dummy[-1]))

        if self.add_latent_gaussian_per_utterance:
            self.state_updates.append((self.platent_utterance_variable_prior, x_reset * self.latent_utterance_variable_prior[-1]))
            self.state_updates.append((self.platent_utterance_variable_approx_posterior, x_reset * self.latent_utterance_variable_approx_posterior[-1]))

            if self.condition_latent_variable_on_dcgm_encoder:
                self.state_updates.append((self.platent_dcgm_avg, x_reset * self.latent_dcgm_avg[-1]))
                self.state_updates.append((self.platent_dcgm_n, x_reset.T * self.latent_dcgm_n[-1]))
        

        # Beam-search variables
        self.beam_x_data = T.imatrix('beam_x_data')
        self.beam_source = T.lvector("beam_source")
        #self.beam_source = T.imatrix("beam_source")
        #         self.x_data = T.imatrix('x_data')
        self.beam_hs = T.matrix("beam_hs")
        self.beam_step_num = T.lscalar("beam_step_num")
        self.beam_hd = T.matrix("beam_hd")
        #self.beam_ran_cost_utterance = T.fvector('beam_ran_cost_utterance')

        # DEBUG CODE USED TO COMPARE TO NO-TRUNCATED VERSION
        #self.get_hs_func = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_cost_mask, self.x_semantic_targets, self.x_reset_mask], outputs=self.hd, updates=self.state_updates, on_unused_input='ignore', name="get_hs_fn")