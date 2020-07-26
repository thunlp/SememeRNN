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

import models

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = 'PATH/TO/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = '.pkl'
V = 1 # version of InferSent

parser.add_argument("--encoder_type", type=str, default='LSTM_baseline', help="see list of encoders")
params, _ = parser.parse_known_args()
encoder_type = params.encoder_type

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


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
    config = {
    'n_words'        :  0          ,
    'word_emb_dim'   :  300   ,
    'enc_lstm_dim'   :  2048   ,
    'n_enc_layers'   :  1   ,
    'dpout_model'    :  0    ,
    'dpout_fc'       :  0       ,
    'fc_dim'         :  512         ,
    'bsize'          :  64     ,
    'n_classes'      :  3      ,
    'nonlinear_fc'   :  1   ,
    'use_cuda'       :  True                  ,
    'sememe_dim'     :  300     ,
    'sememe_size'    :  2186         ,
}
    encoder_types = ['LSTM_baseline', 'LSTM_concat', 'LSTM_gate', 'LSTM_cell', 'LSTM_extra_void', 'LSTM_extra_concat', 'LSTM_extra_gate', 'LSTM_extra_cell',\
                 'BILSTM_baseline', 'BILSTM_concat', 'BILSTM_gate', 'BILSTM_cell', 'BILSTM_extra_void', 'BILSTM_extra_concat', 'BILSTM_extra_gate', 'BILSTM_extra_cell', \
                 'GRU_baseline', 'GRU_concat', 'GRU_gate', 'GRU_cell', 'GRU_extra_void', 'GRU_extra_concat', 'GRU_extra_gate', 'GRU_extra_cell', \
                 'BIGRU_baseline', 'BIGRU_concat', 'BIGRU_gate', 'BIGRU_cell', 'BIGRU_extra_void', 'BIGRU_extra_concat', 'BIGRU_extra_gate', 'BIGRU_extra_cell']
    assert params.encoder_type in encoder_types, "encoder_type must be in " + \ 
                                             str(encoder_types)
    model = eval(models.encoder_type)(config)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)

    params_senteval['infersent'] = model.cuda()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SUBJ', 'MRPC', 'SICKEntailment', 'SICKRelatedness']
    results = se.eval(transfer_tasks)
    print(results)