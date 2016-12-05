#-*- coding:utf-8 -*-
from collections import OrderedDict

def prototype_state():
    state = {} 
     
    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    # Out-of-vocabulary token string
    state['oov'] = '<unk>'
    
    # These are end-of-sequence marks
    state['end_sym_sentence'] = '</s>'

    # Special tokens need to be hardcoded, because model architecture may adapt depending on these
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = 2 # end-of-dialogue symbol </d>
    #state['first_speaker_sym'] = 3 # first speaker symbol <first_speaker>
    #state['second_speaker_sym'] = 4 # second speaker symbol <second_speaker>
    #state['third_speaker_sym'] = 5 # third speaker symbol <third_speaker>
    #state['minor_speaker_sym'] = 6 # minor speaker symbol <minor_speaker>
    #state['voice_over_sym'] = 7 # voice over symbol <voice_over>
    #state['off_screen_sym'] = 8 # off screen symbol <off_screen>
    #state['pause_sym'] = 9 # pause symbol <pause>

    # Training examples will be split into subsequences of size max_grad_steps each.
    # Gradients will be computed on the subsequence, and the last hidden state of all RNNs will
    # be used to initialize the hidden state of the RNNs in the next subsequence.
    state['max_grad_steps'] = 80

    # If this flag is on, the hidden state between RNNs in subsequences is always initialized to zero.
    # Basically, set this flag on if you want to reset all RNN hidden states between 'max_grad_steps' time steps
    
    state['deep_dialogue_input'] = False

    # ----- ACTIVATION FUNCTIONS ---- 
    # Default and recommended setting is: tanh.
    # The dialogue encoder activation function
    state['GRU_rec_activation'] = 'lambda x: T.tanh(x)'
    

    # ----- SIZES ----
    # Dimensionality of hidden layers
    # Dimensionality of (word-level) utterance encoder hidden state
    state['qdim_encoder'] = 512
    # Dimensionality of (word-level) utterance decoder (RNN which generates output) hidden state
    state['qdim_decoder'] = 512
    # Dimensionality of (utterance-level) dialogue hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank word embedding approximation
    state['rankdim'] = 256

    # Dimensionality of Gaussian latent variable (under the assumption that covariance matrix is diagonal)
    state['latent_gaussian_per_utterance_dim'] = 10
    
    # Threshold to clip the gradient
    state['cutoff'] = 1.
    # Learning rate. The rate 0.0001 seems to work well across many tasks.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003

    # Initialization configuration
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = ''
    state['fix_pretrained_word_embeddings'] = False

    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  

    # Sort by length groups of  
    state['sort_k_batches'] = 20
   
    # Modify this in the prototype
    state['save_dir'] = './'
    
    # ----- TRAINING PROCESS -----
    # Frequency of training error reports (in number of batches)
    state['train_freq'] = 10
    # Validation frequency
    state['valid_freq'] = 5000
    # Number of batches to process
    state['loop_iters'] = 1000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    # ----- EVALUATION PROCESS -----
    state['track_extrema_validation_samples'] = True # If set to true will print the extrema (lowest and highest log-likelihood scoring) validation samples
    state['track_extrema_samples_count'] = 100 # Set of extrema samples to track
    state['print_extrema_samples_count'] = 5 # Number of extrema samples to print (chosen at random from the extrema sets)

    state['compute_mutual_information'] = True # If true, the empirical mutural information will be calculcated on the validation set
    
    state['bs'] = 1 #一次只能训练一个数据，目前只支持batch_size=1

    return state

def prototype_zhifu():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "./Data/ttrain.dialogues.pkl" #训练数据
    state['valid_dialogues'] = "./Data/tvalid.dialogues.pkl" #测试数据
    state['dictionary'] = "./Data/ttrain.dict.pkl" #词表
    state['save_dir'] = "./Data/models/"
    state['pretrained_word_embeddings_file'] = './Data/MT_WordEmb.pkl' #预先训练好的词向量
    

    state['max_grad_steps'] = 800 #截断长度
    
    # Handle bleu evaluation
    #state['bleu_evaluation'] = "./tests/bleu/bleu_evaluation"
    #state['bleu_context_length'] = 2
    
    
    # Validation frequency
    state['valid_freq'] = 100000 #每训练多少次进行一次测试并保存模型
    state['loop_iters'] = 2000010 #一个数据算训练一次
    state['train_freq'] = 5000

    # Variables
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'
    
    state['deep_dialogue_input'] = True

    
    state['qdim_encoder'] = 300 #句子向量维度
    # Dimensionality of dialogue hidden layer 
    state['sdim'] = 600 #hs,文本编码的长度
    # Dimensionality of low-rank approximation
    state['rankdim'] = 100 #词向量维度
    state['cnum'] = 5000 #总的标签类别数目
    state['category'] = './Data/category.pkl' #类别标签list
    return state    