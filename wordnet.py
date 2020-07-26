from nltk.corpus import wordnet as wn
import random
hownet_dir = 'dataset/hownet_en.txt'
def read_hownet(hownet_dir):
    dic_hownet = {}
    f1 = open(hownet_dir, 'r')
    line = f1.readline()
    while(line):
        word = line.strip()
        if word not in dic_hownet:
            dic_hownet[word] = []
        line = f1.readline()
        sememes = line.strip().split('\t')
        for item in sememes:
            if item not in dic_hownet[word]:
                dic_hownet[word].append(item)
        line = f1.readline()
    return dic_hownet

hownet = read_hownet(hownet_dir)
new_h = {}
new_s = []
a = [1,2,3,4]
for k in hownet:
    d = []
    for synset in wn.synsets(k):
        for lemma in synset.lemma_names():
            if lemma not in d:
                d.append(lemma)
    if len(d) == 0:
        continue
    random.shuffle(d)
    new_h[k] = d[: random.choice(a)]
f = open('new_hownet.txt', 'w')
for k in new_h:
    f.write(k)
    f.write('\n')
    for kk in new_h[k]:
        f.write(kk)
        f.write('\t')
    f.write('\n')
f.close()
f1 = open('new_sememe.txt', 'w')
for k in new_h:
    for kk in new_h[k]:
        if kk not in new_s:
            f1.write(kk)
            f1.write('\n')
f1.close()
#print(wn.synsets('good')[0].lemma_names())