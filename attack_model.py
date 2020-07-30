import torch 
import numpy as np 

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


class GeneticAtack(object):
    def __init__(self, sess, model, batch_model,
                 neighbour_model,
                 dataset, dist_mat,
                 skip_list,
                 lm,
                 pop_size=20, max_iters=100,
                 n1=20, n2=5,
                 use_lm=True, use_suffix=False):
        self.dist_mat = dist_mat
        self.dataset = dataset
        self.dict = self.dataset.dict
        self.inv_dict = self.dataset.inv_dict
        self.skip_list = skip_list
        self.model = model
        self.batch_model = batch_model
        self.neighbour_model = neighbour_model
        self.sess = sess
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.lm = lm
        self.top_n = n1  # similar words
        self.top_n2 = n2
        self.use_lm = use_lm
        self.use_suffix = use_suffix
        self.temp = 0.3

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, pos, x_cur, x_orig, target, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w and w != 0 else x_cur for w in replace_list]
        new_x_preds = self.neighbour_model.predict(
            self.sess, np.array(new_x_list))

        new_x_scores = new_x_preds[:, target]
        orig_score = self.model.predict(
            self.sess, x_cur[np.newaxis, :])[0, target]
        new_x_scores = new_x_scores - orig_score
        # Eliminate not that clsoe words
        new_x_scores[self.top_n:] = -10000000

        if self.use_lm:
            prefix = ""
            suffix = None
            if pos > 0:
                prefix = self.dataset.inv_dict[x_cur[pos-1]]
            #
            orig_word = self.dataset.inv_dict[x_orig[pos]]
            if self.use_suffix and pos < x_cur.shape[0]-1:
                if (x_cur[pos+1] != 0):
                    suffix = self.dataset.inv_dict[x_cur[pos+1]]
            # print('** ', orig_word)
            replace_words_and_orig = [
                self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in replace_list[:self.top_n]] + [orig_word]
            # print(replace_words_and_orig)
            replace_words_lm_scores = self.lm.get_words_probs(
                prefix, replace_words_and_orig, suffix)
            
            new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
            rank_replaces_by_lm = np.argsort(-new_words_lm_scores)

            filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
            new_x_scores[filtered_words_idx] = -10000000

        if (np.max(new_x_scores) > 0):
            return new_x_list[np.argsort(new_x_scores)[-1]]
        return x_cur

    def perturb(self, x_cur, x_orig, neigbhours, neighbours_dist,  w_select_probs, target):
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(np.sign(w_select_probs)):
            
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        # src_word = x_cur[rand_idx]
        # replace_list,_ =  glove_utils.pick_most_similar_words(src_word, self.dist_mat, self.top_n, 0.5)
        replace_list = neigbhours[rand_idx]
        if len(replace_list) < self.top_n:
            replace_list = np.concatenate(
                (replace_list, np.zeros(self.top_n - replace_list.shape[0])))
        return self.select_best_replacement(rand_idx, x_cur, x_orig, target, replace_list)

    def generate_population(self, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target) for _ in range(pop_size)]

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def attack(self, x_orig, target, max_change=0.4):
        x_adv = x_orig.copy()
        x_len = np.sum(np.sign(x_orig))
        # Neigbhours for every word.
        tmp = [glove_utils.pick_most_similar_words(
            x_orig[i], self.dist_mat, 50, 0.5) for i in range(x_len)]
        neigbhours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]
        neighbours_len = [len(x) for x in neigbhours_list]
        for i in range(x_len):
            if (x_adv[i] < 27):
                # To prevent replacement of words like 'the', 'a', 'of', etc.
                neighbours_len[i] = 0
        w_select_probs = neighbours_len / np.sum(neighbours_len)
        tmp = [glove_utils.pick_most_similar_words(
            x_orig[i], self.dist_mat, self.top_n, 0.5) for i in range(x_len)]
        neigbhours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]
        pop = self.generate_population(
            x_orig, neigbhours_list, neighbours_dist, w_select_probs, target, self.pop_size)
        for i in range(self.max_iters):
            # print(i)
            pop_preds = self.batch_model.predict(self.sess, np.array(pop))
            pop_scores = pop_preds[:, target]
            print('\t\t', i, ' -- ', np.max(pop_scores))
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            logits = np.exp(pop_scores / self.temp)
            select_probs = logits / np.sum(logits)

            if np.argmax(pop_preds[top_attack, :]) == target:
                return pop[top_attack]
            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)

            childs = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size-1)]
            childs = [self.perturb(
                x, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target) for x in childs]
            pop = elite + childs

        return None

