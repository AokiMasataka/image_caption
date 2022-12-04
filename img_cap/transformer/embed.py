import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_length):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.rand(1, max_length, embed_dim))
    
    def forward(self, x):
        return self.embedding[:, x.shape[1]:, :] + x


class ImageEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_size,
        patch_size=16,
        in_channels=3,
        pos_embed=True,
        bias=True
    ):
        super(ImageEmbedding, self).__init__()
        self.image_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=bias
        )
        self.pos_embed = pos_embed
        self.embed_dim = embed_dim
        if pos_embed:
            assert image_size % patch_size == 0
            sequence_length = int((image_size // patch_size) ** 2) 
            self.pos_embedding = nn.Parameter(torch.rand(1, sequence_length, embed_dim))
            self.sequence_length = sequence_length
    
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.image_embedding(x).view(B, self.embed_dim, -1).transpose(1, 2)
        
        if self.pos_embed:
            return x + self.pos_embedding
        else:
            return x


class TextEmbed(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_length, pos_embed=True):
        super(TextEmbed, self).__init__()
        self.text_embed = nn.Embedding(embedding_dim=embed_dim, num_embeddings=vocab_size)
        self.pos_embed = pos_embed
        if pos_embed:
            self.pos_embedding = nn.Parameter(torch.rand(1, max_length, embed_dim))
            self.max_length = max_length
    
    def forward(self, x):
        B, N  = x.shape
        x = self.text_embed(x)
        
        if self.pos_embed:
            return x + self.pos_embedding[:, :N, :]
        else:
            return x


def build_embedding(embedding_config: dict):
    assert isinstance(embedding_config, dict), f'input type is {type(embedding_config)}'
    embedding_modules = {
        'PositionalEmbedding': PositionalEmbedding,
        'ImageEmbedding': ImageEmbedding,
        'TextEmbed': TextEmbed
    }

    _type = embedding_config.pop('type')
    module = embedding_modules[_type]
    return module(**embedding_config)
