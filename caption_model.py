import torch
from torch import nn
from torch.nn import functional
from torchvision import models


class Transformer(nn.Module):
    def __init__(self, cnf, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(dim=cnf.dim)

        self.decoder = Decoder(
            dim=cnf.dim, heads=cnf.heads, depth=cnf.decoder_depth, vocab_size=vocab_size, max_len=cnf.max_seq_len
        )
        self.max_seq_len = cnf.max_seq_len
        self.vocab_size = vocab_size

    def forward(self, image, prov_ids, next_ids, mask):
        memory = self.encoder(image)
        logits = self.decoder(prov_ids, memory, mask)

        logits = logits.view(-1, self.vocab_size)
        next_ids = next_ids.view(-1)
        return functional.cross_entropy(logits, next_ids, ignore_index=0)

    @torch.inference_mode()
    def inference(self, image, device='cpu'):
        memory = self.encoder(image)
        sequence = torch.full((1, 1), fill_value=2, dtype=torch.long, device=device)

        for _ in range(self.max_seq_len):
            logit = self.decoder(sequence, memory, mask=None)
            logit = logit[:, -1, :].argmax(dim=1)
            sequence = torch.cat((sequence, logit.unsqueeze(0)), dim=1)
            if logit.item() == 3:
                break
        return sequence


class Encoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.1):
        super(Encoder, self).__init__()
        self.features = models.convnext_tiny(pretrained=True, drop_path_rate=drop_path_rate).features
        self.head = nn.Sequential(
            nn.Dropout(p=0.125),
            nn.Linear(in_features=768, out_features=dim, bias=True)
        )

    def forward(self, x):
        x = self.features(x).flatten(2)
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, heads, depth, vocab_size, max_len, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = PositionalEmbedding(dim, max_seq_len=max_len)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim, heads, drop_prob=drop_prob, attention_drop_prob=drop_prob) for _ in range(depth)
        ])
        self.logit = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, memory, mask):
        x = self.embed(x)
        x = self.pos_embed(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, mask)
        return self.logit(x)


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, drop_prob=0.1, attention_drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(dim, heads, drop_prob=attention_drop_prob)
        self.attention = MultiHeadAttention(dim, heads, drop_prob=attention_drop_prob)
        self.feed_forward = FeedForward(dim, dim_ff=dim * 4, drop_prob=drop_prob)

    def forward(self, x, memory, mask=None, cache=None):
        x = self.masked_attention(q=x, k=x, v=x, mask=mask)
        x = self.attention(q=x, k=memory, v=memory, mask=None)
        x = self.feed_forward(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, drop_prob=0.1):
        super().__init__()
        self.dim = dim
        self.d_k = dim // heads
        self.h = heads
        self.drop_prob = drop_prob

        self.q_linear = nn.Linear(dim, dim, bias=False)
        self.v_linear = nn.Linear(dim, dim, bias=False)
        self.k_linear = nn.Linear(dim, dim, bias=False)
        self.fc = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-8)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        residual = q
        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        q = torch.matmul(q, k.transpose(-2, -1)) / self.d_k
        if mask is not None:
            mask = mask.unsqueeze(1)
            q = q.masked_fill(mask == 0, -1e9)
        q = functional.softmax(q, dim=-1)   # mask
        q = functional.dropout(q, p=self.drop_prob, training=self.training)
        q = torch.matmul(q, v).transpose(1, 2).contiguous().view(bs, -1, self.dim)
        q = functional.dropout(self.fc(q), p=self.drop_prob, training=self.training)
        return self.norm(q + residual)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_ff=2048, drop_prob=0.1):
        super().__init__()
        self.w_1 = nn.Linear(dim, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.drop_prob = drop_prob

    def forward(self, x):
        residual = x
        x = functional.relu(self.w_1(x), inplace=True)
        x = functional.dropout(self.w_2(x), p=self.drop_prob, training=self.training)
        return self.norm(x + residual)


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super(PositionalEmbedding, self).__init__()
        self.pos_embed = nn.Parameter(torch.rand(max_seq_len, dim))

    def forward(self, x):
        return self.pos_embed[:x.shape[1], :] + x
