import pandas as pd
import numpy as np
from collections import defaultdict


emb_dir = '/data/dchaudhu/ESWC_challenge/Embeddings/'
word_embeddings = ['embeddings_snap_s512_e15.txt',
                       'embeddings_snap_s128_e15.txt', 'embeddings_snap_s128_e30.txt', 'embeddings_snap_s128_e50.txt', 'embeddings_snap_s512_e50.txt', 'embeddings_snap_s512_e30.txt']

for emb_file in word_embeddings:
    emb_d_f = emb_file.split('.')[0] + '.npy'
    emb_dim = int(emb_file.split('_')[2].split('s')[1])
    emb_vec = defaultdict(float)
    with open(emb_dir + emb_file, 'r') as f:
        emb = f.readlines()
    print ("Loadded the vector, creating the dictionary.....")
    for j in range(1, len(emb)):
        word = emb[j].split('\n')[0].strip().split()[0]
        try:
            vec = np.array(emb[j].split('\n')[0].strip().split()[1:]).astype(np.float32)
        except ValueError:
            continue
        # try:
        if len(vec) == emb_dim:
            emb_vec[word] = vec
        else:
            continue
    np.save(emb_dir + emb_d_f, emb_vec)