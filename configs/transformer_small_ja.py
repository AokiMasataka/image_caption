IMAGE_SIZE = 256
BATCH_SIZE = 128
MAX_LENGTH = 196

model=dict(
    tokenizer='cl-tohoku/bert-base-japanese',
    vocab_size=32768,
    init_config=dict(
        pretrained=None,
        encoder_weight='./works/exp/pretrained/encoder_small.pth',
        decoder_weight=None,
        encoder_embed_weight='./works/exp/pretrained/encoder_embed_small.pth',
        decoder_embed_weight=None,
    ),
    encoder_config=dict(
        embed_dim=192,
        num_heads=3,
        depth=12,
        qkv_bias=False,
        attn_drop=0.1,
        drop=0.1,
        eps=1e-6
    ),
    decoder_config=dict(
        embed_dim=192,
        num_heads=3,
        depth=12,
        qkv_bias=False,
        pos_bias=False,
        attn_drop=0.1,
        drop=0.1,
        eps=1e-6
    ),
    encoder_embed_config=dict(
        type='ImageEmbedding',
        embed_dim=192,
        image_size=IMAGE_SIZE,
        patch_size=16,
        in_channels=3,
        pos_embed=True
    ),
    decoder_embed_config=dict(
        type='TextEmbed',
        embed_dim=192,
        max_length=MAX_LENGTH,
        pos_embed=True
    ),
    meshed=False,
)

data=dict(
    data_path='./stair_captions/stair_captions_train_data.txt',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
)


train_config=dict(
    base_lr=1e-4,
    use_amp=False,
    device='cuda',
    epochs=8,
    log_step=100,
    work_dir='./works/small'
)
