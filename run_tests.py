import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from data import Amazon_loader
from model import CNN
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(
    description='Amazon Runner'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--h_dim', type=int, default=100, metavar='',
                    help='hidden dimension (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--emb_drop', type=float, default=0.3, metavar='',
                    help='embedding dropout (default: 0.3)')
parser.add_argument('--mb_size', type=int, default=128, metavar='',
                    help='size of minibatch (default: 128)')
parser.add_argument('--n_epoch', type=int, default=500, metavar='',
                    help='number of iterations (default: 500)')
parser.add_argument('--randseed', type=int, default=123, metavar='',
                    help='random seed (default: 123)')
parser.add_argument('--no_tqdm', default=False, action='store_true',
                    help='disable tqdm progress bar')


args = parser.parse_args()

# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.gpu:
    torch.cuda.manual_seed(args.randseed)

max_seq_len = 100

amazon = Amazon_loader(dom='Toys_Games')
model = CNN(amazon.emb_dim, amazon.vocab_size, h_dim=args.h_dim, gpu=args.gpu)

solver = optim.Adam(model.parameters(), lr=args.lr)

if args.gpu:
    model.cuda()

if __name__ == '__main__':
    for epoch in range(args.n_epoch):
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')
        model.train()
        train_iter = enumerate(amazon.get_iter('train'))

        for it, mb in train_iter:
            review, y = mb
            print(review, y)
            output = model(review)

            loss = F.binary_cross_entropy_with_logits(output, y)

            loss.backward()
            #clip_gradient_threshold(model, -10, 10)
            solver.step()
            solver.zero_grad()
