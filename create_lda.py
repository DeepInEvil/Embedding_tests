import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models
import gc
import sys
import os.path
import re
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#get stopwords
stop = stopwords.words('english')
stop.append('\\)')
stop.append('\\(')

lda_model = './lda_models/amazon_lda'
lda_dict = './lda_models/amazon_dict'

if __name__ == '__main__':
    all_reviews = np.array(pd.read_table('reviews.txt', sep=',', header=None))[:, 0]
    all_reviews = [[word for word in sent if word not in stop] for sent in all_reviews]
    #create amazon dictionary
    print ('Done loading the data...')
    amzn_dict = gensim.corpora.Dictionary(all_reviews)
    amzn_dict.save(lda_dict)
    bow_dat = [amzn_dict.doc2bow(sent) for sent in all_reviews]
    #create lda model
    print ('Creating the LDA model...')
    lda = models.LdaMulticore(bow_dat, id2word=amzn_dict, num_topics=50, workers=30, passes=1)
    lda.save(lda_model)





