# https://openreview.net/pdf?id=BJe932EYwS

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .transformer_base import MultiHeadAttention
from .transformer import Decoder, VisionTransformer


class PositionPredictor(nn.Module):
    def __init__(self, dim, heads, depth, drop_prob=0.1):
        super(PositionPredictor, self).__init__()
        self.pos_pred_list = nn.ModuleList([
            MultiHeadAttention(dim=dim, heads=heads, drop_prob=drop_prob) for _ in range(depth)
        ])

    def forward(self, x):
        for pos_pred in self.pos_pred_list:
            x = pos_pred(x, x, x)
        return x


class Bridge(nn.Module):
    def __init__(self, dim):
        super(Bridge, self).__init__()
        self.logit = nn.Linear(in_features=dim, out_features=16)

    def forward(self, x):
        batch, _, _ = x.shape
        repeater = self.logit(x).argmax(dim=2)

        query = []
        for i in range(batch):
            sequence = []
            for j, rep in enumerate(repeater[i]):
                sequence.append(x[i, j].unsqueeze(0).repeat(rep, 1))
            query.append(torch.cat(sequence, dim=0))
        query = pad_sequence(query, batch_first=True)

        return query


class PositionNonAutoregressiveTransformer(nn.Module):
    def __init__(self, cnf, vocab_size, max_seq_len):
        super(PositionNonAutoregressiveTransformer, self).__init__()
        self.encoder = VisionTransformer(
            image_size=cnf.image_size,
            patch_size=cnf.patch_size,
            dim=cnf.dim,
            depth=cnf.encoder_depth,
            heads=cnf.heads,
            drop_prob=cnf.drop_prob
        )

        self.decoder = Decoder(
            dim=cnf.dim, heads=cnf.heads, depth=cnf.decoder_depth, vocab_size=vocab_size, max_len=cnf.max_seq_len
        )
        self.decoder.embed = nn.Identity()
        self.decoder.pos_embed = nn.Identity()

        self.position_predictor = PositionPredictor(dim=cnf.dim, heads=cnf.heads, depth=4, drop_prob=cnf.drop_prob)
        self.bridge = Bridge(dim=cnf.dim)
        self.max_seq_len = max_seq_len

    def forward(self, images, target=None, mask=None):
        memory = self.encoder(images)

        x = self.bridge(memory)
        x = x[:, :self.max_seq_len, :]
        pos = self.position_predictor(x)

        x = self.decoder(x + pos, memory, mask=None)
        return x
