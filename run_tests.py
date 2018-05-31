import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from data import Amazon_loader
from model import CNN
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from gensim import models
import gensim


parser = argparse.ArgumentParser(
    description='Amazon Runner'
)

parser.add_argument('--gpu', default=True, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--h_dim', type=int, default=100, metavar='',
                    help='hidden dimension (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--emb_drop', type=float, default=0.3, metavar='',
                    help='embedding dropout (default: 0.3)')
parser.add_argument('--mb_size', type=int, default=64, metavar='',
                    help='size of minibatch (default: 128)')
parser.add_argument('--n_epoch', type=int, default=100, metavar='',
                    help='number of iterations (default: 500)')
parser.add_argument('--randseed', type=int, default=666, metavar='',
                    help='random seed (default: 666)')
parser.add_argument('--no_tqdm', default=False, action='store_true',
                    help='disable tqdm progress bar')


args = parser.parse_args()

# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.gpu:
    torch.cuda.manual_seed(args.randseed)

max_seq_len = 100
root_dir = '/data/dchaudhu/ESWC_challenge/'
lda_model = models.LdaModel.load(root_dir + 'lda_models/amazon_lda')
lda_dict =  gensim.corpora.Dictionary.load(root_dir + '/lda_models/amazon_dict')
#w2i = np.load(root_dir+'/data/vocab.npy').item()
#i2w = {v: k for k, v in w2i.items()}


def get_id2word(idx, idx2w_dict):
    """
    get id2word mappings
    :param idx:
    :param idx2w_dict:
    :return:
    """
    try:
        return idx2w_dict[idx]
    except KeyError:
        return 'UNK'


def get_lda_vec(lda_dict):
    """
    get lda vector
    :param lda_dict:
    :return:
    """
    lda_vec = np.zeros(50, dtype='float32')
    for id, val in lda_dict:
        lda_vec[id] = val
    return lda_vec


def get_theta(texts, lda, dictionari, idx2word):
    """
    get doc-topic distribution vector for all reviews
    :param texts:
    :param lda:e
    :param dictionari:
    :param idx2word:
    :return:
    """
    #texts = np.transpose(texts)
    texts = texts.data.cpu().numpy()
    texts = [[get_id2word(idx, idx2word) for idx in sent] for sent in texts]
    review_alphas = np.array([get_lda_vec(lda[dictionari.doc2bow(sentence)]) for sentence in texts])
    return torch.from_numpy(review_alphas)


def evaluate(model, dataset, mode):

    model.eval()
    data_iter = dataset.get_iter(mode)

    acc = 0.0
    c = 0.0

    for mb in data_iter:
        review, y = mb
        topic = Variable(get_theta(review, lda_model, lda_dict, i2w)).cuda()
        output = F.sigmoid(model(review, topic))

        scores_o = output.data.cpu() if args.gpu else output.data
        #print (scores_o)
        acc = acc + (roc_auc_score(y.data.cpu().numpy(), scores_o.numpy()))
        c = c + 1

    return (acc/c)


def test(model, dataset, mode='test'):
    model.eval()
    data_iter = dataset.get_iter(mode)
    test_scores = []
    for mb in data_iter:
        review = mb
        topic = Variable(get_theta(review, lda_model, lda_dict, i2w)).cuda()
        output = F.sigmoid(model(review, topic))
        test_scores.append(output.cpu().data.numpy())
    return (np.concatenate(test_scores))


def run_model(amazon, model, solver):
    early_stop = []
    for epoch in range(args.n_epoch):
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('------------------------------/home/deep/attentions-------------')
        model.train()
        train_iter = enumerate(amazon.get_iter('train'))

        for it, mb in train_iter:
            review, y = mb
            topic = Variable(get_theta(review, lda_model, lda_dict, i2w)).cuda()
            #print (topic.size())
            output = model(review, topic)

            loss = F.binary_cross_entropy_with_logits(output, y)

            loss.backward()
            #clip_gradient_threshold(model, -10, 10)
            solver.step()
            solver.zero_grad()

        acc = evaluate(model, amazon, 'valid')
        print("Accuracy after epoch:" + str(epoch) + " is " + str(acc))
        if epoch < 30:
            early_stop.append(acc)
        elif acc < np.max(early_stop[:(epoch-7)]):
            print("Exiting training......")
            break
        else:
            early_stop.append(acc)

    #acc_test = evaluate(model, amazon, 'valid')
    #print("Accuracy on test_set:" + str(acc_test))
    return acc


if __name__ == '__main__':
    domains = ['Video_Games', 'Books', 'Toys_Games', 'Tools_Home_Improvement', 'Amazon_Instant_Video', 'Movies_TV', 'Electronics', 'Health',
               'Shoes', 'Baby', 'Automotive', 'Software', 'Sports_Outdoors', 'Clothing_Accessories', 'Beauty', 'Patio', 'Music',
               'Pet_Supplies', 'Office_Products', 'Home_Kitchen']
    #word_embeddings = ['embeddings_snap_s256_e15.txt', 'embeddings_snap_s256_e50.txt', 'embeddings_snap_s256_e30.txt',
                         # 'embeddings_snap_s512_e15.txt', 'embeddings_snap_s128_e15.txt', 'embeddings_snap_s128_e30.txt',
                         # 'embeddings_snap_s128_e50.txt', 'embeddings_snap_s512_e50.txt', 'embeddings_snap_s512_e30.txt']
    # word_embeddings = ['embeddings_snap_s512_e15.txt', 'embeddings_snap_s128_e15.txt', 'embeddings_snap_s128_e30.txt',
    #                    'embeddings_snap_s128_e50.txt', 'embeddings_snap_s512_e50.txt', 'embeddings_snap_s512_e30.txt']
    word_embeddings = ['embeddings_snap_s512_e50']
    perf_dict = {}
    for emb in word_embeddings:

        print ("Running model for embedding" + str(emb))
        emb_dim = int(emb.split('_')[2].split('s')[1])
        amazon = Amazon_loader(emb_file=emb, emb_dim=emb_dim)
        w2i = np.load(root_dir + '/data/vocab.npy').item()
        i2w = {v: k for k, v in w2i.items()}
        global i2w
        #amazon = Amazon_loader(dom=root_dir+domain, emb_dim=300)
        model = CNN(amazon.emb_dim, amazon.vocab_size, h_dim=args.h_dim, pretrained_emb=amazon.vectors,
                    gpu=args.gpu)

        solver = optim.Adam(model.parameters(), lr=args.lr)

        if args.gpu:
            model.cuda()

        perf_dict[emb+':'+'amazonWE'] = run_model(amazon, model, solver)
        test_out = test(model, amazon)
        np.save(root_dir + 'test_out'+emb+'npy', test_out)
    for k, v in perf_dict.items():
        print (k, v)
        print ("===============")




