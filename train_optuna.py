import os
import time
import argparse
import random
import torch
import numpy as np
import cv2
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_crohme_dataset
from models.can import CAN
from training import train, eval

def train_test_CAN_model(params=None):
    if params is None:
        params = dict(experiment='CAN', epochs=240, batch_size=8, workers=0, 
                      train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                      
                      optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                      eps='1e-6', weight_decay='1e-4', beta=0.9, 

                      dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                      
                      train_image_path='datasets/train_images.pkl', train_label_path='datasets/train_labels.txt',
                      eval_image_path='datasets/test_images.pkl', eval_label_path='datasets/test_labels.txt',
                      word_path='datasets/word.txt', 
                      
                      collate_fn='collate_fn', 
                      densenet={'ratio': 16, 'nDenseBlocks': 8, 'growthRate': 8, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True},
                      encoder={'input_channel': 1, 'out_channel': 116}, 
                      decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 256, 'hidden_size': 256}, 
                      counting_decoder={'in_channel': 116, 'out_channel': 22}, 
                      attention={'attention_dim': 128, 'word_conv_kernel': 1}, 

                      attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                      whiten_type='None', max_step=256,
                      optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', data_augmentation=2, log_dir='logs')
        
    from models.densenet import DenseNet
    model_temp = DenseNet(params=params)

    a = torch.zeros((1, 1, 150, 150))
    out = model_temp(a)

    print(out.shape[1])

    # get the output channels parameter
    params['encoder']['out_channel'] = out.shape[1]
    params['counting_decoder']['in_channel'] = out.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    print('Using device', device)

    train_loader, eval_loader = get_crohme_dataset(params)

    model = CAN(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

    print(model.name)
    model = model.to(device)

    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))
    
    max_eval_expRate = 0
    min_step = 0
    max_train_expRate = 0

    for epoch in range(params['epochs']):
        train_loss, train_word_score, train_expRate = train(params, model, optimizer, epoch, train_loader, writer=None)

        if (epoch+1) >= 5 and ((epoch+1) % 5 == 0):
            eval_loss, eval_word_score, eval_expRate = eval(params, model, epoch, eval_loader, writer=None)

            if eval_expRate > max_eval_expRate:
                max_eval_expRate = eval_expRate
                max_train_expRate = train_expRate
                min_step = epoch

            # stop if no improvement for more than 30 epochs
        if (epoch+1) >= min_step + 30:
            break

    return max_eval_expRate, max_train_expRate

    #CONTINUAR DAQUI

if __name__ == '__main__':

    train_test_CAN_model()