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

import yaml
import argparse

def train_test_CAN_model(params=None, base_name=""):
    if params is None:
        params = dict(experiment='CAN', epochs=500, batch_size=8, workers=0, 
                      train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                      
                      optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                      eps='1e-6', weight_decay='1e-4', beta=0.9, 

                      dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                      
                      train_image_path='datasets/optuna/train_image.pkl', train_label_path='datasets/optuna/train_labels.txt',
                      eval_image_path='datasets/optuna/test_image.pkl', eval_label_path='datasets/optuna/test_labels.txt',
                      word_path='datasets/word.txt', 
                      
                      collate_fn='collate_fn', 
                      densenet={'ratio': 16, 'nDenseBlocks': 4, 'growthRate': 24, 'reduction': 0.5, 'bottleneck': False, 'use_dropout': True},
                      encoder={'input_channel': 1, 'out_channel': 684}, 
                      decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 256, 'hidden_size': 256}, 
                      counting_decoder={'in_channel': 684, 'out_channel': 20}, 
                      attention={'attention_dim': 512, 'word_conv_kernel': 1}, 

                      attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                      whiten_type='None', max_step=256,
                      optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', data_augmentation=100, log_dir='logs')
        
    from models.densenet import DenseNet
    model_temp = DenseNet(params=params)

    a = torch.zeros((1, 1, 150, 150))
    out = model_temp(a)

    print(out.shape[1])

    # get the output channels parameter
    params['encoder']['out_channel'] = out.shape[1]
    params['counting_decoder']['in_channel'] = out.shape[1]

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    params['device'] = device_type
    print('Using device', device)

    train_loader, eval_loader = get_crohme_dataset(params)

    model = CAN(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    #model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'
    model.name = base_name + str(now)

    print(model.name)
    model = model.to(device)

    os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    config_path = os.path.join(params['checkpoint_dir'], model.name, model.name) + '.yaml'
    with open(config_path, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
    #os.system(f'cp {params} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')

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

                save_checkpoint(model, optimizer, eval_word_score, eval_expRate, epoch+1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])

            # stop if no improvement for more than 30 epochs
        if epoch >= min_step + 100:
            break

    return max_eval_expRate, max_train_expRate

    #CONTINUAR DAQUI

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=int, required=True)
    parser.add_argument("--hme7k", action='store_true')
    parser.add_argument("--reduced", action='store_true')
    args = parser.parse_args()

    print("Configuração: " + str(args.config))
    print("Base HME7K: " + str(args.hme7k))
    print("Reduzir dimensao para 150: " + str(args.reduced))

    

    if args.hme7k and args.reduced:
        sufix = "_150"
    else:
        sufix = ""

    if args.hme7k:
        train_image='datasets/hme7k/train_image' + sufix + '.pkl'
        train_label='datasets/hme7k/train_labels.txt'
        eval_image='datasets/hme7k/test_image' + sufix + '.pkl'
        eval_label='datasets/hme7k/test_labels.txt'
        word='datasets/word_hme7k.txt'
        out=19
        augmentation=0

        dir_name = "hme7k_optuna_" + str(args.config) + sufix + "_"

    else:
        train_image='datasets/optuna/train_image.pkl'
        train_label='datasets/optuna/train_labels.txt'
        eval_image='datasets/optuna/test_image.pkl'
        eval_label='datasets/optuna/test_labels.txt'
        word='datasets/word.txt'
        out=20
        augmentation=100

        dir_name = "soma_subtr_optuna_" + str(args.config) + "_"

    print(dir_name)

    if args.config == 1:    
        params = dict(experiment='CAN', epochs=500, batch_size=8, workers=0, 
                        train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                        
                        optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                        eps='1e-6', weight_decay='1e-4', beta=0.9, 

                        dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                        
                        train_image_path=train_image, train_label_path=train_label,
                        eval_image_path=eval_image, eval_label_path=eval_label,
                        word_path=word, 
                        
                        collate_fn='collate_fn', 
                        densenet={'ratio': 16, 'nDenseBlocks': 16, 'growthRate': 24, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True},
                        encoder={'input_channel': 1, 'out_channel': 684}, 
                        decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 64, 'hidden_size': 64}, 
                        counting_decoder={'in_channel': 684, 'out_channel': out}, 
                        attention={'attention_dim': 512, 'word_conv_kernel': 1}, 

                        attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                        whiten_type='None', max_step=256,
                        optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', data_augmentation=augmentation, log_dir='logs')
    
    elif args.config == 2:
        params = dict(experiment='CAN', epochs=500, batch_size=8, workers=0, 
                        train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                        
                        optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                        eps='1e-6', weight_decay='1e-4', beta=0.9, 

                        dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                        
                        train_image_path=train_image, train_label_path=train_label,
                        eval_image_path=eval_image, eval_label_path=eval_label,
                        word_path=word, 
                        
                        collate_fn='collate_fn', 
                        densenet={'ratio': 16, 'nDenseBlocks': 4, 'growthRate': 24, 'reduction': 0.5, 'bottleneck': False, 'use_dropout': True},
                        encoder={'input_channel': 1, 'out_channel': 684}, 
                        decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 256, 'hidden_size': 256}, 
                        counting_decoder={'in_channel': 684, 'out_channel': out}, 
                        attention={'attention_dim': 512, 'word_conv_kernel': 1}, 

                        attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                        whiten_type='None', max_step=256,
                        optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', data_augmentation=augmentation, log_dir='logs')
    
    elif args.config == 3:
        params = dict(experiment='CAN', epochs=500, batch_size=8, workers=0, 
                        train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                        
                        optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                        eps='1e-6', weight_decay='1e-4', beta=0.9, 

                        dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                        
                        train_image_path=train_image, train_label_path=train_label,
                        eval_image_path=eval_image, eval_label_path=eval_label,
                        word_path=word, 
                        
                        collate_fn='collate_fn', 
                        densenet={'ratio': 16, 'nDenseBlocks': 16, 'growthRate': 24, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True},
                        encoder={'input_channel': 1, 'out_channel': 684}, 
                        decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 64, 'hidden_size': 64}, 
                        counting_decoder={'in_channel': 684, 'out_channel': out}, 
                        attention={'attention_dim': 256, 'word_conv_kernel': 1}, 

                        attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                        whiten_type='None', max_step=256,
                        optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', data_augmentation=augmentation, log_dir='logs')
        
    elif args.config == 4:
        params = dict(experiment='CAN', epochs=500, batch_size=8, workers=0, 
                        train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                        
                        optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                        eps='1e-6', weight_decay='1e-4', beta=0.9, 

                        dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                        
                        train_image_path=train_image, train_label_path=train_label,
                        eval_image_path=eval_image, eval_label_path=eval_label,
                        word_path=word, 
                        
                        collate_fn='collate_fn', 
                        densenet={'ratio': 16, 'nDenseBlocks': 16, 'growthRate': 24, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True},
                        encoder={'input_channel': 1, 'out_channel': 684}, 
                        decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 128, 'hidden_size': 128}, 
                        counting_decoder={'in_channel': 684, 'out_channel': out}, 
                        attention={'attention_dim': 512, 'word_conv_kernel': 1}, 

                        attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                        whiten_type='None', max_step=256,
                        optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', data_augmentation=augmentation, log_dir='logs')
        
    elif args.config == 5:
        params = dict(experiment='CAN', epochs=500, batch_size=8, workers=0, 
                        train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                        
                        optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                        eps='1e-6', weight_decay='1e-4', beta=0.9, 

                        dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                        
                        train_image_path=train_image, train_label_path=train_label,
                        eval_image_path=eval_image, eval_label_path=eval_label,
                        word_path=word, 
                        
                        collate_fn='collate_fn', 
                        densenet={'ratio': 16, 'nDenseBlocks': 8, 'growthRate': 24, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True},
                        encoder={'input_channel': 1, 'out_channel': 684}, 
                        decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 64, 'hidden_size': 64}, 
                        counting_decoder={'in_channel': 684, 'out_channel': out}, 
                        attention={'attention_dim': 512, 'word_conv_kernel': 1}, 

                        attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                        whiten_type='None', max_step=256,
                        optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', data_augmentation=augmentation, log_dir='logs')
    
    train_test_CAN_model(params, dir_name)
