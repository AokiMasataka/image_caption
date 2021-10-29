import torch
from torch import nn
from .transformer_base import EncoderLayer, Encoder, Decoder, PositionalEmbedding, VisionEmbedding


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, drop_prob, length_pred=False):
        super(VisionTransformer, self).__init__()
        seq_len = (image_size // patch_size) ** 2
        seq_len = seq_len + 1 if length_pred else seq_len
        self.length_pred = length_pred

        self.image_embed = VisionEmbedding(dim, patch_size)
        self.pos_embed = PositionalEmbedding(dim, seq_len)
        if self.length_pred:
            self.query = nn.Parameter(torch.rand(1, 1, dim))

        self.encoder_layer = nn.Sequential(
            *[EncoderLayer(dim, heads, drop_prob=drop_prob, attention_drop_prob=drop_prob) for _ in range(depth)]
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.image_embed(x)
        if self.length_pred:
            x = torch.cat((self.query.repeat(batch, 1, 1), x), dim=1)
        x = self.pos_embed(x)
        x = self.encoder_layer(x)
        return x


class Transformer(nn.Module):
    def __init__(self, cnf, vocab_size):
        super(Transformer, self).__init__()
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

    def forward(self, image, target, mask=None):
        memory = self.encoder(image)
        return self.decoder(target, memory, mask)

    def forward_ones(self, image, max_seq_len):
        b = image.shape[0]
        memory = self.encoder(image)
        seq = torch.full((b, 1), fill_value=2, dtype=torch.long, device='cuda')

        for i in range(max_seq_len):
            logit = self.decoder(seq, memory, mask=None)
            logit = logit[:, -1, :].argmax(1).view(-1, 1)
            seq = torch.cat((seq, logit), dim=1)
        return seq


class Parallel(nn.Module):
    def __init__(self, cnf, vocab_size):
        super(Parallel, self).__init__()
        self.encoder = VisionTransformer(
            image_size=cnf.image_size,
            patch_size=cnf.patch_size,
            dim=cnf.dim,
            depth=cnf.encoder_depth,
            heads=cnf.heads,
            drop_prob=cnf.drop_prob,
            length_pred=False
        )

        self.decoder = Decoder(
            dim=cnf.dim, heads=cnf.heads, depth=cnf.decoder_depth, vocab_size=vocab_size, max_len=cnf.max_seq_len
        )

        self.length_predict = nn.Linear(in_features=cnf.dim, out_features=cnf.max_seq_len)
        self.max_seq_len = cnf.max_seq_len

    def forward(self, image, target=None, mask=None):
        memory = self.encoder(image)
        length, memory = memory[:, 0, :], memory[:, 1:, :]
        length = self.length_predict(length).argmax(dim=1)

        if target is None:
            target = torch.zeros((length.shape[0], self.max_seq_len), dtype=torch.long, device='cuda')
            for i in range(target):
                target[i, :length[i]] = 4

        return self.decoder(target, memory)


# [PAD] 0
# [UNK] 1
# [CLS] 2
# [SEP] 3
# [MASK] 4
