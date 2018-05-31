import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNN(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=300, pretrained_emb=None, gpu=False, emb_drop=0.5, pad_idx=0, top_dim=50):
        super(CNN, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.n_filter = h_dim // 3
        self.h_dim = self.n_filter * 3 + top_dim
        self.out_h = 100
        self.conv3 = nn.Conv2d(1, self.n_filter, (3, emb_dim))
        self.conv4 = nn.Conv2d(1, self.n_filter, (4, emb_dim))
        self.conv5 = nn.Conv2d(1, self.n_filter, (5, emb_dim))

        self.emb_drop = nn.Dropout(emb_drop)
        self.fc = nn.Parameter(nn.init.xavier_normal(torch.FloatTensor(self.h_dim, self.out_h)))
        #self.fc2 = nn.Parameter(nn.init.xavier_normal(torch.FloatTensor(self.out_h + top_dim, 1)))
        self.b = nn.Parameter(torch.FloatTensor([0]))

        if gpu:
            self.cuda()

    def forward(self, x, top=None):
        emb_x = self.emb_drop(self.word_embed(x))
        out = self._forward(emb_x, top)

        return out

    def _forward(self, x, top):
        x = x.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim

        x3 = F.relu(self.conv3(x)).squeeze()
        x4 = F.relu(self.conv4(x)).squeeze()
        x5 = F.relu(self.conv5(x)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        out = torch.cat([x3, x4, x5, top], dim=-1)
        #fc1 = torch.mm(out, self.fc)
        #o = torch.mm(torch.cat([fc1, top], dim=-1), self.fc2) + self.b
        o = torch.mm(out, self.fc)

        return o.squeeze()
