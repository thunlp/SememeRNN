import numpy as np
import glove_utils
import pickle
import torch
from models import LSTM_gate

parser.add_argument("--encoder_type", type=str, default='LSTM_baseline', help="see list of encoders")
params, _ = parser.parse_known_args()

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
#change model path if needed
MODEL_PATH = '.pkl'
model = eval(models.encoder_type)(config)
model.load_state_dict(torch.load(MODEL_PATH))

adversary = EntailmentAttack(model, dist_mat, pop_size=128, max_iters=12, n1=5)

success_count = 0
for i in range(TEST_SIZE):
    print('\n')
    test_idx = test_idxs[i]
    attack_input = [test[0][test_idx][np.newaxis,:], test[1][test_idx][np.newaxis,:]]
    if np.sum(np.sign(attack_input[1])) < 10:
        continue
    attack_pred = np.argmax(model.predict(attack_input))
    true_label = np.argmax(test[2][test_idx])
    if attack_pred != true_label:
        print('Wrong classified')
    else:
        if true_label == 2:
            target = 0
        elif true_label == 0:
            target = 2
        else:
            target = 0 if np.random.uniform() < 0.5 else 2
        start_time = time()
        attack_result = adversary.attack(attack_input, target)
        if attack_result is None:
            print('**** Attack failed **** ')
        else:
            success_count += 1
            print('***** DONE ', len(test_list) , '------' )
            visulaize_result(model, attack_input, attack_result)
            test_times.append(time()-start_time)
        test_list.append(test_idx)
        input_list.append(attack_input)
        output_list.append(attack_result)

print(success_count / len(test_list))