import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, sememe):
        self.dictionary = Dictionary()
        self.sememe = sememe
        if 'wikitext' in path:
            self.train, self.train_sememes = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid, self.valid_sememes = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test, self.test_sememes = self.tokenize(os.path.join(path, 'test.txt'))
        else:
            self.train, self.train_sememes = self.tokenize(os.path.join(path, 'ptb.train.txt'))
            self.valid, self.valid_sememes = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
            self.test, self.test_sememes = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            sememes = []
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    sememes.append(self.sememe.read_word_sememe(word))
                    token += 1

        return ids, sememes
