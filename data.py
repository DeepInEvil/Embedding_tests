import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import gensim
import os
emb_dir = '/data/dchaudhu/ESWC_challenge/Embeddings/'

class Amazon_loader:
    '''
    For loading amazon words
    '''
    def __init__(self, home='/data/dchaudhu/ESWC_challenge/data/', positive='positive.tsv', negative='negative.tsv', batch_size=64,
                 max_seq_len=100, gpu=True, emb_file=None, emb_dim=100):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.gpu = gpu
        self.pos = np.array(pd.read_table(home + positive, sep='\t', header=None)[1])
        self.neg = np.array(pd.read_table(home + negative, sep='\t', header=None)[1])
        self.dat = np.concatenate((self.pos, self.neg))
        self.y = np.concatenate((np.ones(500000), np.zeros(500000)))
        self.X_te = np.array(pd.read_table(home+'test.tsv', sep='\t', header=None)[1])
        self.X_tr, self.X_val, self.y_tr, self.y_val = self.create_train_test((self.dat, self.y), split_size=0.1)
        #self.X_tr, self.X_val, self.y_tr, self.y_val = self.create_train_test((self.X_tr, self.y_tr), split_size=0.125)

        if (os.path.exists(home+'vocab.npy')):
            self.vocab = np.load(home+'vocab.npy').item()
            print ("Loaded the vocabulary of size:" + str(len(self.vocab.keys())))
        else:
            self.vocab = self.create_vocab(self.X_tr)
            np.save(home + 'vocab.npy', self.vocab)
            print("Created vocabulary of size:" + str(len(self.vocab.keys())))
        #print (self.X_tr[0])
        self.X_tr = [[self.getW2Id(self.vocab, w) for w in sent.split()] for sent in self.X_tr if not isinstance(sent, float)]
        self.X_te = [[self.getW2Id(self.vocab, w) for w in sent.split()] for sent in self.X_te if not isinstance(sent, float)]
        self.X_val = [[self.getW2Id(self.vocab, w) for w in sent.split()] for sent in self.X_val if not isinstance(sent, float)]
        self.train, self.test, self.valid = {}, {}, {}
        self.train['X'] = self.X_tr
        self.train['y'] = self.y_tr
        self.test['X'] = self.X_te
        self.valid['X'] = self.X_val
        self.valid['y'] = self.y_val

        self.emb_dim = emb_dim
        self.vocab_size = len(self.vocab) + 1
        i2w = {v: k for k, v in self.vocab.items()}

        emb_d_f = emb_file.split('.')[0] + '.npy'
        if os.path.exists(emb_dir+emb_d_f):
            print ('loading the embeddng dictionary from file')
            emb_vec = np.load(emb_dir+emb_d_f).item()
        else:
            emb_vec = {}
            with open(emb_dir+emb_file, 'r') as f:
                emb = f.readlines()
            for j in range(1, len(emb)):
                word = emb[j].split('\n')[0].strip().split()[0]
                vec = emb[j].split('\n')[0].strip().split()[1:]
                #try:
                if len(vec) == self.emb_dim:
                    emb_vec[word] = vec
                else:
                    continue
            np.save(emb_dir+emb_d_f, emb_vec)
            del emb
            print('Saving embedding dictionary')

        #emb = gensim.models.KeyedVectors.load_word2vec_format('/data/dchaudhu/ESWC_challenge/Embeddings/'
        #                                                  'GoogleNews-vectors-negative300.bin', binary=True)
        vectors = np.zeros((self.vocab_size, self.emb_dim))
        #emb = np.array(pd.read_csv('/data/dchaudhu/ESWC_challenge/Embeddings/sentic2vec.csv', encoding="cp1252"))

        for i in i2w.keys():
            try:
                vectors[i] = emb_vec[i2w[i]]
            except KeyError:
                continue

        #print(self.vocab['video'], vectors[self.vocab['video']], emb_vec['video'])
        '''
        for j in range(len(emb)):
            word = emb[j][0]
            vec = emb[j][1:]
            try:
                vectors[self.vocab[word]] = vec
            except Exception:
                continue
        '''
        del emb_vec
        self.vectors = torch.from_numpy(vectors.astype(np.float32))

    def create_vocab(self, train):
        "create word to id mappings"
        vocab = defaultdict(float)
        out_vocab = []
        for sent in train:
            if isinstance(sent, float):
                continue
            else:
                for w in sent.split():
                    vocab[w] += 1.0
        for k, v in vocab.items():
            if v > 5.0:
                out_vocab.append(k)
        w2i = dict(zip(out_vocab, range(1, len(out_vocab) + 1)))
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

    def get_iter(self, mode='train'):
        if mode == 'train':
            dataset = self.train
        elif mode == 'test':
            dataset = self.test
        elif mode == 'valid':
            dataset = self.valid

        if mode not in 'test':
            for i in range(0, len(dataset['y']), self.batch_size):
                reviews = dataset['X'][i:i+self.batch_size]
                y = dataset['y'][i:i+self.batch_size]

                reviews, y = self._load_batch(reviews, y, self.batch_size)

                yield reviews, y
        else:
            for i in range(0, len(dataset['X']), self.batch_size):
                reviews = dataset['X'][i:i + self.batch_size]
                #y = dataset['y'][i:i + self.batch_size]

                reviews, y = self._load_batch(reviews, self.batch_size, test=True)

                yield reviews

    def _load_batch(self, reviews, y=None, size=32, test=False):
        review_arr = np.zeros([size, self.max_seq_len], np.int)
        y_arr = np.zeros(size, np.float32)

        if not test:
            for j, (rvw, y_r) in enumerate(zip(reviews, y)):
                rvw = rvw[:self.max_seq_len]
                review_arr[j, :len(rvw)] = rvw
                y_arr[j] = float(y_r)

            review = Variable(torch.from_numpy(review_arr))
            y = Variable(torch.from_numpy(y_arr))

            if self.gpu:
                review, y = review.cuda(), y.cuda()
            return review, y
        else:
            for j, rvw in enumerate(reviews):
                rvw = rvw[:self.max_seq_len]
                review_arr[j, :len(rvw)] = rvw
                #y_arr[j] = float(y_r)

            review = Variable(torch.from_numpy(review_arr))
            #y = Variable(torch.from_numpy(y_arr))

            if self.gpu:
                review = review.cuda()
            return review
