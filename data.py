import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import gensim


class Amazon_loader:
    '''
    For loading amazon words
    '''
    def __init__(self, dom='Home_Kitchen', positive='positive.tsv', negative='negative.tsv', batch_size=64,
                 max_seq_len=100, gpu=True, emb_file=None, emb_dim=100):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.gpu = gpu
        self.pos = np.array(pd.read_table(dom + '/' + positive, sep = '\t', header=None)[1])
        self.neg = np.array(pd.read_table(dom + '/' + negative, sep = '\t', header=None)[1])
        self.dat = np.concatenate((self.pos, self.neg))
        self.y = np.concatenate((np.ones(25000), np.zeros(25000)))
        self.X_tr, self.X_te, self.y_tr, self.y_te = self.create_train_test((self.dat, self.y), split_size=0.2)
        self.X_tr, self.X_val, self.y_tr, self.y_val = self.create_train_test((self.X_tr, self.y_tr), split_size=0.125)
        self.vocab = self.create_vocab(self.X_tr)
        np.save(dom + '/' + 'vocab.npy', self.vocab)
        self.X_tr = [[self.getW2Id(self.vocab, w) for w in sent.split()] for sent in self.X_tr if not isinstance(sent, float)]
        self.X_te = [[self.getW2Id(self.vocab, w) for w in sent.split()] for sent in self.X_te if not isinstance(sent, float)]
        self.X_val = [[self.getW2Id(self.vocab, w) for w in sent.split()] for sent in self.X_val if not isinstance(sent, float)]
        self.train, self.test, self.valid = {}, {}, {}
        self.train['X'] = self.X_tr
        self.train['y'] = self.y_tr
        self.test['X'] = self.X_te
        self.test['y'] = self.y_te
        self.valid['X'] = self.X_val
        self.valid['y'] = self.y_val

        self.emb_dim = emb_dim
        self.vocab_size = len(self.vocab)
        i2w = {v: k for k, v in self.vocab.items()}
        print (self.vocab, self.X_tr[0])
        '''
        if emb_file is not None:
            with open(emb_file, 'r') as f:
                emb = f.readlines()
        '''
        emb = gensim.models.KeyedVectors.load_word2vec_format('/data/dchaudhu/ESWC_challenge/Embeddings/'
                                                          'GoogleNews-vectors-negative300.bin', binary=True)
        vectors = np.zeros((self.vocab_size, self.emb_dim))

        '''
        for j in range(1, len(emb)):
            word = emb[j].split('\n')[0].strip().split()[0]
            vec = emb[j].split('\n')[0].strip().split()[1:]
            try:
                self.vectors[self.vocab[word]] = vec
            except Exception:
                continue
        '''
        for i in i2w.keys():
            try:
                vectors[i] = emb[i2w[i]]
            except KeyError:
                continue
        del emb
        self.vectors = torch.from_numpy(vectors.astype(np.float32))

    def create_vocab(self, train):
        "create word to id mappings"
        vocab = defaultdict(float)
        for sent in train:
            if isinstance(sent, float):
                continue
            else:
                for w in sent.split():
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

    def create_train_test(self, data, split_size):
        "create splits"
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test

    def get_iter(self, dataset='train'):
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'test':
            dataset = self.test
        elif dataset == 'valid':
            dataset = self.valid

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