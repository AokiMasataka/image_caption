import torch
from torch import nn
from torch.nn import functional


class Decoder(nn.Module):
    def __init__(self, dim, heads, depth, vocab_size, max_len, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = PositionalEmbedding(dim, max_seq_len=max_len)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim, heads, drop_prob=drop_prob, attention_drop_prob=drop_prob) for _ in range(depth)
        ])
        self.logit = nn.Linear(dim, vocab_size)

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


class Encoder(nn.Module):
    def __init__(self, dim, heads, depth, vocab_size, max_len, drop_prob=0.1):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = PositionalEmbedding(dim, max_seq_len=max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim, heads, drop_prob=drop_prob, attention_drop_prob=drop_prob) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, drop_prob=0.1, attention_drop_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads, drop_prob=attention_drop_prob)
        self.feed_forward = FeedForward(dim, dim_ff=dim * 4, drop_prob=drop_prob)

    def forward(self, x, mask=None):
        x = self.attention(q=x, k=x, v=x, mask=mask)
        return self.feed_forward(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, drop_prob=0.1):
        super().__init__()
        self.dim = dim
        self.d_k = dim // heads
        self.h = heads
        self.drop_prob = drop_prob

        self.q_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop_prob)
        self.out = nn.Linear(dim, dim)
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
        q = functional.dropout(self.out(q), p=self.drop_prob, training=self.training)
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


class VisionEmbedding(nn.Module):
    def __init__(self, dim, patch_size):
        super(VisionEmbedding, self).__init__()
        patch_t2 = (patch_size, patch_size)
        self.vision_embed = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_t2, stride=patch_t2)

    def forward(self, image):
        return self.vision_embed(image).flatten(2).transpose(1, 2)


@torch.jit.script
def mish(x):
    return x * torch.tanh(functional.softplus(x))
