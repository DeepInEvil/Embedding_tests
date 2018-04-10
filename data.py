import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable


class Amazon_loader:
    '''
    For loading amazon words
    '''
    def __init__(self, dom='Home_Kitchen', positive='positive.tsv', negative='negative.tsv', batch_size=64, max_seq_len=100, gpu=True):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.gpu = gpu
        self.pos = np.array(pd.read_table(dom + '/' + positive, sep = '\t', header=None)[1])
        self.neg = np.array(pd.read_table(dom + '/' + negative, sep = '\t', header=None)[1])
        self.dat = np.concatenate((self.pos, self.neg))
        self.y = np.concatenate((np.ones(25000), np.zeros(25000)))
        self.X_tr, self.X_te, self.y_tr, self.y_te = self.create_train_test((self.dat, self.y))
        self.vocab = self.create_vocab(self.X_tr)
        np.save(dom + '/' + 'vocab.npy', self.vocab)
        self.X_tr = [[self.getW2Id(self.vocab, w) for w in sent] for sent in self.X_tr]
        self.X_te = [[self.getW2Id(self.vocab, w) for w in sent] for sent in self.X_te]
        self.train, self.test = {}, {}
        self.train['X'] = self.X_tr
        self.train['y'] = self.y_tr
        self.test['X'] = self.X_te
        self.test['y'] = self.y_te

        self.emb_dim = 200
        self.vocab_size = len(self.vocab)

    def create_vocab(self, train):
        "create word to id mappings"
        vocab = defaultdict(float)
        for sent in train:
            for w in sent:
                vocab[w] += 1.0

        w2i = dict(zip(vocab.keys(), range(1, len(vocab) + 1)))
        w2i['UNK'] = len(w2i) + 1
        return w2i

    def getW2Id(self, w2i, word):
        "get ids for words"
        try:
            return w2i[word]
        except KeyError:
            return w2i['UNK']

    def create_train_test(self, data):
        "create splits"
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test

    def get_iter(self, dataset='train'):
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'test':
            dataset = self.test

        for i in range(0, len(dataset['y']), self.batch_size):
            reviews = dataset['X'][i:i+self.batch_size]
            y = dataset['y'][i:i+self.batch_size]

            reviews, y = self._load_batch(reviews, y, self.batch_size)

            yield reviews, y

    def _load_batch(self, reviews, y, size):
        review_arr = np.zeros([size, self.max_seq_len], np.int)
        y_arr = np.zeros(size, np.float32)

        for j, (rvw, y_r) in enumerate(zip(reviews, y)):
            rvw = rvw[:self.max_seq_len]
            review_arr[j, :len(rvw)] = rvw
            y_arr[j] = float(y_r)

        review = Variable(torch.from_numpy(review_arr))
        y = Variable(torch.from_numpy(y_arr))

        if self.gpu:
            review, y = review.cuda(), y.cuda()

        return review, y