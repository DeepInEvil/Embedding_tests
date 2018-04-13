import pandas as pd
import numpy as np


emb_dir = '/data/dchaudhu/ESWC_challenge/Embeddings/'
word_embeddings = ['embeddings_snap_s512_e15.txt',
                       'embeddings_snap_s128_e15.txt', 'embeddings_snap_s128_e30.txt', 'embeddings_snap_s128_e50.txt', 'embeddings_snap_s512_e50.txt', 'embeddings_snap_s512_e30.txt']

for emb_file in word_embeddings:
    emb_d_f = emb_file.split('.')[0] + '.npy'
    emb_dim = int(emb.split('_')[2].split('s')[1])
    emb_vec = {}
    with open(emb_dir + emb_file, 'r') as f:
        emb = f.readlines()
    for j in range(1, len(emb)):
        word = emb[j].split('\n')[0].strip().split()[0]
        vec = emb[j].split('\n')[0].strip().split()[1:]
        # try:
        if len(vec) == emb_dim:
            emb_vec[word] = vec
        else:
            continue
    np.save(emb_dir + emb_d_f, emb_vec)