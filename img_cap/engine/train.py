import os
import sys
import logging
import torch
from torch.cuda import amp
from transformers import BertTokenizer
from transformers import logging as transformer_logging
from ..transformer import Transformer
from ..dataset import build_loader
from ..utils import AvgManager, set_logger


def train(config: dict):
    transformer_logging.set_verbosity_error()
    assert isinstance(config, dict)

    train_config = config['train_config']
    model_config = config['model']
    data_config = config['data']

    os.makedirs(train_config['work_dir'], exist_ok=True)

    set_logger(log_file=os.path.join(train_config['work_dir'], 'train.log'))

    logging.info(f'Python info: {sys.version}')
    logging.info(f'PyTroch version: {torch.__version__}')
    logging.info(f'GPU model: {torch.cuda.get_device_name(0)}\n')

    tokenizer = BertTokenizer.from_pretrained(model_config.pop('tokenizer'))
    model = Transformer(**model_config)
    data_loader = build_loader(
        data_path=data_config['data_path'],
        image_size=data_config['image_size'],
        batch_size=data_config['batch_size'],
        max_length=data_config['max_length'],
        tokenizer=tokenizer,
        shuffle=True
    )
    logging.info(msg='deploy model')
    logging.info(msg=f'number of datas: {data_loader.dataset.__len__()}')

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['base_lr'])
    model.to(train_config['device'])
    scaler = amp.GradScaler(enabled=train_config['use_amp'])
    
    logging.info(msg='start trainning')
    for epoch in range(train_config['epochs']):
        loss_train = AvgManager()
        for step, (images, input_ids) in enumerate(data_loader, start=1):
            images, input_ids = images.to(train_config['device']), input_ids.to(train_config['device'])

            optimizer.zero_grad()
            with amp.autocast(enabled=train_config['use_amp']):
                loss = model.forward_train(images=images, input_ids=input_ids)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_train.update(value=loss.item())

            if step % train_config['log_step'] == 0:
                logging.info(msg=f'epoch: {epoch} - step: {step} - loss train: {loss_train():.4f}')
                loss_train.__init__()
        
        save_dir = os.path.join(train_config['work_dir'], f'epoch{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        logging.info(msg=f'epoch: {epoch} - save to {os.path.join(save_dir, f"epoch{epoch}")}')
        model.save_pretrained(save_dir)


if __name__ == '__main__':
    train()
