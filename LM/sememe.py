 #coding: utf-8 
import random
class Sememe(object):
    def __init__(self, hownet_dir, sememe_dir, lemma_dir, filename=None, lower=True, drop_rate = 1):
        #self.idxToLabel = {}
        self.labelToIdx = {}
        self.lower = lower
        self.lemma_dict = self.read_lemmatization(lemma_dir)
        hownet_ori = self.read_hownet(hownet_dir)
        if drop_rate < 1:
            self.hownet = {k: v for k,v in hownet_ori.items() if random.uniform(0, 1) <= drop_rate}
        else:
            self.hownet = hownet_ori
        if filename is not None:
            self.loadFile(filename)
 
    def size(self):
        return len(self.labelToIdx)

    # Load entries from a file.
    def loadFile(self, file_dir):
        f = open(file_dir, 'r')
        line = f.readline()
        while(line):
            line = f.readline()
            a = line.strip().split('\t')
            for item in a:
                if item not in self.labelToIdx:
                    self.labelToIdx[item] = len(self.labelToIdx)
            
            line = f.readline()
        print(len(self.labelToIdx))

    def getIndex(self, key):
        if key in self.labelToIdx:
            return self.labelToIdx[key]
        else:

            return None

    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels):
        vec = []
        vec += [self.getIndex(label) for label in labels]
        return vec

    '''
    input: a list of each word in a single sentence
    output: a list of each word's sememe list in a single sentence
    '''
    def read_sememe(self, labels):
        sentence = []
        sentence += [self.read_word_sememe(label) for label in labels]

        return sentence


    '''
    input: a word
    output: a list of the word's sememe
    '''
    def read_word_sememe(self, word):
        labels = []
        if word in self.hownet:
            for item in self.hownet[word]:
                if self.getIndex(item) not in labels:
                    labels.append(self.getIndex(item))
        elif word in self.lemma_dict:
            if self.lemma_dict[word] in self.hownet:
                for item in self.hownet[self.lemma_dict[word]]:
                    if self.getIndex(item) not in labels:
                        labels.append(self.getIndex(item))

        return labels

    def read_hownet(self, hownet_dir):
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

    def read_lemmatization(self, lemma_dir):
        dic_lemma = {}
        for line in open(lemma_dir):
            line = line.strip().split()
            dic_lemma[line[1]] = line[0]
        return dic_lemma
