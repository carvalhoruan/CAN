import logging
import sys
import optuna

import numpy as np
import time

from train_optuna import train_test_CAN_model

work_dir = ''
#log_file_name = 'optimize_HME_100k_reduced_CAN'
log_file_name = 'optimize_Base_Soma_Subtr_CAN'

def evaluation_trial(trial):
    model = 'CAN'

    if model == 'CAN':
        decoder_input_size = trial.suggest_categorical('decoder_input_size', [64, 128, 256]) # default: 256

        params = dict(experiment='CAN', epochs=500, batch_size=8, workers=0, 
                      train_parts=1, valid_parts=1, valid_start=0, save_start=0, 
                      
                      optimizer='Adadelta', lr=1, lr_decay='cosine', step_ratio=10, step_decay=5, 
                      eps='1e-6', weight_decay='1e-4', beta=0.9, 

                      dropout=True, dropout_ratio=0.5, relu=True, gradient=100, gradient_clip=True, use_label_mask=False, 
                      
                      train_image_path='datasets/optuna/train_image.pkl', train_label_path='datasets/optuna/train_labels.txt',
                      eval_image_path='datasets/optuna/test_image.pkl', eval_label_path='datasets/optuna/test_labels.txt',
                      word_path='datasets/word.txt', 
                      
                      collate_fn='collate_fn', 
                      densenet={'ratio': 16,
                                'nDenseBlocks': trial.suggest_categorical('nDenseBlocks', [4, 8, 16]),  # default: 16
                                'growthRate': trial.suggest_categorical('growthRate', [8, 16, 24]), # default: 24,
                                'reduction': trial.suggest_categorical('reduction', [0.1, 0.2, 0.5]), # default: 0.5
                                'bottleneck': trial.suggest_categorical('bottleneck', [True, False]), # default: True
                                'use_dropout': trial.suggest_categorical('use_dropout', [True, False]) # default: True
                                },
                      encoder={'input_channel': 1, 'out_channel': 116}, 
                      decoder={'net': 'AttDecoder', 'cell': 'GRU', 'input_size': decoder_input_size, 'hidden_size': decoder_input_size}, 
                      counting_decoder={'in_channel': 116, 'out_channel': 22}, 
                      attention={'attention_dim': trial.suggest_categorical('attention_dim', [128, 256, 512]), # default: 512
                                 'word_conv_kernel': 1}, 

                      attention_map_vis_path='vis/attention_map', counting_map_vis_path='vis/counting_map', 
                      whiten_type='None', max_step=256,
                      optimizer_save=False, finetune=False, checkpoint_dir='checkpoints', 
                      data_augmentation=trial.suggest_categorical('data_augmentation', [10, 100]), # default 0
                      log_dir='logs')
        
        print(params)

        trial.set_user_attr("params", params)

        test_exp_rate, train_exp_rate = train_test_CAN_model(params=params)

        trial.set_user_attr("train_exp_rate", train_exp_rate)

    return test_exp_rate


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = log_file_name
storage_name = "sqlite:///{}.db".format(work_dir + study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize',
                            sampler=optuna.samplers.TPESampler())

print('Trials:', len(study.trials))

study.optimize(evaluation_trial, n_trials=100)

print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)