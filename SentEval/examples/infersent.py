# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
from sememe import Sememe
import argparse

parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--model", type=str, default='BILSTM_baseline', help="NLI data path (SNLI or MultiNLI)")
params, _ = parser.parse_known_args()
MODEL_PATH = []
for dir in os.listdir('/data1/private/qinyujia/Sememe-enhanced-RNN-qin/savedir'):
    if params.model in dir and 'encoder' in dir:
        if 'BI' not in params.model:
            if 'BI' not in dir:
                MODEL_PATH.append('/data1/private/qinyujia/Sememe-enhanced-RNN-qin/savedir/' + dir)
        else:
            if 'BI' in dir:
                MODEL_PATH.append('/data1/private/qinyujia/Sememe-enhanced-RNN-qin/savedir/' + dir)
#MODEL_PATH = '/data1/private/qinyujia/Sememe-enhanced-RNN-qin/savedir/' + 'model.pickle_BIGRU_baseline_9791.encoder.pkl'

# get models.py from InferSent repo
from models import InferSent
from models_s import *

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = '../../glove/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
#MODEL_PATH = 'encoder/infersent1.pkl'
#MODEL_PATH = '/data1/private/qinyujia/Sememe-enhanced-RNN-qin/savedir/model.pickle_BILSTM_baseline_2465.encoder.pkl'
V = 1 # version of InferSent

#assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), 'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

#sememe.txt是共有哪些英文sememe
sememe_dir = '/data1/private/qinyujia/Sememe-enhanced-RNN-qin/dataset/sememe.txt'
#hownet_en.txt是每个英文词有哪些英文sememe
hownet_dir = '/data1/private/qinyujia/Sememe-enhanced-RNN-qin/dataset/hownet_en.txt'
#用来将每个词还原到原型
lemma_dir =  '/data1/private/qinyujia/Sememe-enhanced-RNN-qin/dataset/lemmatization.txt'
sememe = Sememe(hownet_dir = hownet_dir, sememe_dir = sememe_dir, lemma_dir = lemma_dir, filename = hownet_dir, lower = True, meaningless= False, wordnet = False)

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    config_nli_model = {
    'word_emb_dim'   :  300   ,
    'enc_lstm_dim'   :  2048   ,
    'sememe_dim'     :  300     ,
    'sememe_size'    :  2186
    }
    #model = InferSent(params_model)
    model = eval(params.model)(config_nli_model, sememe)
    #model = BILSTM_baseline(config_nli_model, sememe)
    for model_path in MODEL_PATH:
        model.load_state_dict(torch.load(model_path))
        model.set_w2v_path(PATH_TO_W2V)

        

        params_senteval['infersent'] = model.cuda()
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        transfer_tasks = ['MRPC', 'CR']
        #transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                        #'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                        #'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                        #'Length', 'WordContent', 'Depth', 'TopConstituents',
                        #'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                        #'OddManOut', 'CoordinationInversion']
        results = se.eval(transfer_tasks)
        print(results)
