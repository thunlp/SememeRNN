import numpy as np
import glove_utils
import pickle
import torch
from models import LSTM_gate

class EntailmentAttack(object):
    def __init__(self, model, dist_mat, pop_size=4, max_iters=10, n1=8, n2=4):
        self.model = model
        self.dist_mat = dist_mat
        self.n1 = n1
        self.n2 = n2
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.temp = 1.0

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, target, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, target) for _ in range(pop_size)]

    def perturb(self, x_cur, x_orig, neighbours_list, w_select_probs, target):
        rand_idx = np.random.choice(
            w_select_probs.shape[0], 1, p=w_select_probs)[0]
        # while x_cur[rand_idx] != x_orig[rand_idx]:
        #    rand_idx = np.random.choice(x_cur.shape[0], 1, p=w_select_probs)[0]
        new_w = np.random.choice(neighbours_list[rand_idx])
        return self.do_replace(x_cur, rand_idx, new_w)

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def attack(self, x_orig, target):
        x1_adv = x_orig[0].copy().ravel()
        x2_adv = x_orig[1].copy().ravel()
        x1_orig = x_orig[0].ravel()
        x2_orig = x_orig[1].ravel()
        x1_len = np.sum(np.sign(x1_adv))
        x2_len = np.sum(np.sign(x2_adv))
        tmp = [glove_utils.pick_most_similar_words(x2_adv[i], self.dist_mat, 50, 0.5) if x2_adv[i] != 0 else ([], [])
               for i in range(len(x2_adv))]
        neighbours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]
        neigbhours_len = [len(x) for x in neighbours_list]
        w_select_probs = neigbhours_len / np.sum(neigbhours_len)
        tmp = [glove_utils.pick_most_similar_words(x2_adv[i], self.dist_mat, self.n1, 0.5) if x2_adv[i] != 0 else ([], [])
               for i in range(len(x2_adv))]
        neighbours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]

        pop = np.array(self.generate_population(
            x2_adv, neighbours_list, w_select_probs, target, self.pop_size))
        pop = pop.reshape(self.pop_size, -1)
        # print(pop)
        pop_x1 = np.tile(x1_adv, (self.pop_size, 1, 1)
                         ).reshape(self.pop_size, -1)
        for iter_idx in range(self.max_iters):
            pop_preds = self.model.predict([pop_x1, pop])
            pop_scores = pop_preds[:, target]
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]
            if np.argmax(pop_preds[top_attack, :]) == target:
                return x1_orig, pop[top_attack]
            print(iter_idx, ' : ', np.max(pop_scores))
            logits = np.exp(pop_scores / self.temp)
            pop_select_probs = logits / np.sum(logits)

            elite = [pop[top_attack]]
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=pop_select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=pop_select_probs)

            childs = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size-1)]
            childs = [self.perturb(
                x, x2_orig, neighbours_list, w_select_probs, target) for x in childs]
            pop = elite + childs
            pop = np.array(pop)
        return None

def visulaize_result(model, attack_input, attack_output, file):
    str_labels = ['Contradiction', 'neutral', 'entailment']
    orig_pred = model.predict(attack_input)
    adv_pred = model.predict([attack_output[0][np.newaxis,:], attack_output[1][np.newaxis,:]])
    file.write(reconstruct(attack_output[0].ravel(), inv_vocab) , ' || ', reconstruct(attack_output[1].ravel(), inv_vocab))

TEST_SIZE = 500
model = LSTM_gate(config)
adversary = EntailmentAttack(model, dist_mat, pop_size=128, max_iters=12, n1=5)
success_count = 0
f1 = open('adversarial_example.txt', 'r')
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
            visulaize_result(model, attack_input, attack_result, f1)
            test_times.append(time()-start_time)
        test_list.append(test_idx)
        input_list.append(attack_input)
        output_list.append(attack_result)

print(success_count / len(test_list))