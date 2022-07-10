from asyncio import sleep
from click import Tuple
import torch
import numpy as np
import random
import cv2
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torchvision
import torchsummary
from torch.optim import Adam
from torch import optim
import torch.backends.cudnn as cudnn

from sklearn import metrics
from sklearn.metrics import recall_score, accuracy_score, precision_score, log_loss, classification_report, f1_score
from metrics.metric import calculate_cls_metrics

from utils.Log import Logger
from utils.EarlyStopping import EarlyStopping
from utils.ModelSaver import ModelSaver
from utils.util import is_refined_model

from dataloader.gen_dataloader import *

import sys, os
import os.path as osp
sys.path.append(osp.dirname(__file__))

from loss.focal_loss import FocalLoss as FL
from loss.weightedBCE_loss import WeightedBinaryCrossEntropy as WBCE

from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import UndefinedMetricWarning

from module.train_torch import calculate_metric, define_log_writer, define_optimizer, define_device, define_criterion, find_current_earlystopping_score, save_result
from module.train_kfold import define_log_writer_for_kfold, define_log_writer_for_kfold_pairwise
from dataloader.KFold import CustomizeKFold
from loss.contrastive_loss import ContrastiveLoss


###################################################################
################# SINGLE CNN FOR RGB IMAGE STREAM #################
###################################################################

def eval_kfold_twooutput_image_stream_stream(model ,dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0 ):
    loss = 0
    mac_accuracy = 0
    model.eval()
    y_label = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Push to device
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs, labels = inputs.float().to(device), labels.long().to(device)

            # Forward network
            output = model.forward(inputs)

            # Loss in a batch
            batch_loss = criterion(output, labels)
            # Cumulate into running val loss
            loss += batch_loss.item()

            # Find accuracy
            values, preds = torch.max(output, dim=1)
            mean_acc = torch.mean((labels.data == preds), dtype=torch.float32).item()
            mac_accuracy += mean_acc
            #
            pred_label = preds.cpu().detach().numpy()
            y_pred_label.extend(pred_label)
            
    assert len(y_label) == len(y_pred_label), "Bug"
    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros

def train_kfold_twooutput_image_stream(model_, what_fold='all', n_folds=5, use_trick=True, criterion_name=None, train_dir = '', val_dir ='', test_dir = '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=20, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="xception", args_txt="", augmentation=True, gpu_id=2):
   
    kfold = CustomizeKFold(n_folds=n_folds, train_dir=train_dir, val_dir=val_dir, trick=use_trick)
    next_fold=False
    # what_fold: 'x', 'x-all', 'all', 'x-only'
    if 'all' not in what_fold:
        if 'only' not in what_fold:
            try:
                fold_resume = int(what_fold)
            except:
                raise Exception("what fold should be an integer")
        else:
            fold_resume = int(what_fold.split('_')[0])
    else:
        if 'all' == what_fold:
            fold_resume = 0
        else:
            fold_resume = int(what_fold.split('_')[0])

    import copy
    model_copy = copy.deepcopy(model_)
    for fold_idx in range(n_folds):
        print("\n*********************************************************************************************")
        print("****************************************** FOLD {} *******************************************".format(fold_idx))
        print("*********************************************************************************************")
        if fold_idx < fold_resume:
            continue
        if 'only' in what_fold and fold_idx > fold_resume:
            continue
        # Generate dataloader train and validation:

        trainset, valset = kfold.get_fold(fold_idx=fold_idx)

        # Generate dataloader train and validation 
        dataloader_train, dataloader_val, num_samples = generate_dataloader_single_cnn_stream_for_kfold(train_dir, trainset, valset, image_size, batch_size, num_workers, augmentation=augmentation)
        dataloader_test = generate_test_dataloader_single_cnn_stream_for_kfold(test_dir, image_size, batch_size, num_workers)
        
        # Define optimizer (Adam) and learning rate decay
        init_lr = lr
        init_epoch = 0
        init_step = 0
        init_global_acc = 0
        init_global_loss = 0
        if resume != "":
            try:
                if 'epoch' in checkpoint:
                    init_epoch = int(resume.split('_')[3])
                    init_step = init_epoch * len(dataloader_train)
                    init_lr = lr * (0.8 ** ((init_epoch - 1) // 2))
                    print('Resume epoch: {} - with step: {} - lr: {}'.format(init_epoch, init_step, init_lr))
                if 'step' in checkpoint:
                    init_step = int(resume.split('_')[3])
                    init_epoch = int(init_step / len(dataloader_train))
                    init_lr = lr * (0.8 ** (init_epoch // 2))
                    with open(osp.join(checkpoint, 'global_acc_loss.txt'), 'r') as f:
                        line = f.read().strip()
                        init_global_acc = float(line.split(',')[0])
                        init_global_loss = float(line.split(',')[1])
                    print('Resume step: {} - in epoch: {} - lr: {} - global_acc: {} - global_loss: {}'.format(init_step, init_epoch, init_lr, init_global_acc, init_global_loss))               
            except:
                pass
            
        model = copy.deepcopy(model_copy)
        optimizer = define_optimizer(model, model_name=model_name, init_lr=init_lr, division_lr=division_lr, use_pretrained=use_pretrained)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2*i for i in range(1, epochs//2 + 1)], gamma = 0.8)
        
        # Define devices
        device = define_device(seed=seed, model_name=model_name)

        # Define criterion
        criterion = define_criterion(criterion_name, num_samples)
        criterion = criterion.to(device)
        
        # Define logging factor:
        ckc_pointdir, log, batch_writer, epoch_writer_tup, step_writer_tup = define_log_writer_for_kfold(checkpoint, fold_idx, resume, args_txt, (model, model_name, image_size))
        epoch_ckcpoint, epoch_val_writer, epoch_test_writer = epoch_writer_tup
        step_ckcpoint, step_val_writer, step_test_writer = step_writer_tup
            
        # Define Early stopping and Model saver
        early_stopping = EarlyStopping(patience=es_patience, verbose=True, tunning_metric=es_metric)
        epoch_model_saver = ModelSaver(save_metrics=["val_loss", 'test_acc'])
        step_model_saver = ModelSaver(save_metrics=["val_loss", 'test_acc'])
        
        # Define and load model
        model = model.to(device)
        if resume != "":
            model.load_state_dict(torch.load(osp.join(checkpoint, resume)))
        model.train()

        # print(model.base_net[0].init_block.conv1.conv.weight.data)
        # exit(0)

        running_loss = 0
        running_acc = 0

        global_loss = init_global_loss
        global_acc = init_global_acc
        global_step = init_step

        for epoch in range(init_epoch, epochs):
            print("\n=========================================")
            print("Epoch: {}/{}".format(epoch+1, epochs))
            print("Model: {} - {}".format(model_name, args_txt))
            print("lr = ", optimizer.param_groups[0]['lr'])

            # Train
            model.train()
            print("Training...")
            for inputs, labels in tqdm(dataloader_train):
                # print("inputs: ", inputs[0])
                # print("labels: ", labels)
                # if global_step == 10:
                #     exit(0)
                global_step += 1
                # Push to device
                inputs, labels = inputs.to(device).float(), labels.to(device).long()
                # Clear gradient after a step
                optimizer.zero_grad()

                # Forward netword
                output = model(inputs)   # Shape (32, 2)

                # Find loss
                loss = criterion(output, labels)
                # print("logps: ", logps, "     ====     loss: ", loss)
                
                # Backpropagation and update weights
                loss.backward()
                optimizer.step()

                # update running (train) loss and accuracy
                running_loss += loss.item()
                global_loss += loss.item()
                values, preds = torch.max(output, dim=1)
                running_acc += torch.mean(torch.tensor(labels.data == preds, dtype=torch.float32).cpu().detach()).item()
                global_acc += torch.mean(torch.tensor(labels.data == preds, dtype=torch.float32).cpu().detach()).item()

                # Save step's loss:
                # To tensorboard and to writer
                log.write_scalar(scalar_dict={"Loss/Single step": loss.item()}, global_step=global_step)
                batch_writer.write("{},{:.4f}\n".format(global_step, loss.item()))

                # Eval after <?> iters:
                if eval_per_iters != -1:
                    if (global_step % eval_per_iters == 0):
                        model.eval()
                        # Eval validation set
                        val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_kfold_twooutput_image_stream_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                        
                        # Eval test set
                        test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_kfold_twooutput_image_stream_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_test_writer, log, global_step, global_loss/global_step, global_acc/global_step, test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=False, phase="test")
                        # Save model:
                        step_model_saver(global_step, [val_loss, test_mic_acc], step_ckcpoint, model)
                        step_model_saver.save_last_model(step_ckcpoint, model, global_step)
                        step_model_saver.save_model(step_ckcpoint, model, global_step, save_ckcpoint=False, global_acc=global_acc, global_loss=global_loss)
                    
                        es_cur_score = find_current_earlystopping_score(es_metric, val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2])
                        early_stopping(es_cur_score)
                        if early_stopping.early_stop:
                            print('Early stopping. Best {}: {:.6f}'.format(es_metric, early_stopping.best_score))
                            time.sleep(5)
                            os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, args_txt if resume == '' else '', "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], 'fold' if resume == '' else 'resume', fold_idx)))
                            next_fold = True
                        if next_fold:
                            break
                        model.train()
            if next_fold:
                break
        if next_fold:
            next_fold=False
            continue
                
            # Reset to the next epoch
            running_loss = 0
            running_acc = 0
            scheduler.step()
            model.train()

        # Sleep 5 seconds for rename ckcpoint dir:
        time.sleep(5)
        # Save epoch acc val, epoch acc test, step acc val, step acc test
        if osp.exists(ckc_pointdir):
            os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, args_txt if resume == '' else '', "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], 'fold' if resume == '' else 'resume', fold_idx)))
    return