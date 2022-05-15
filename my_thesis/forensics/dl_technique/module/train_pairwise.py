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
from.train_torch import *
from loss.contrastive_loss import ContrastiveLoss

#############################################
################# DUAL STREAM
#############################################
def eval_pairwise_dual_stream(model, dataloader, device, bce_loss, contrastive_loss, adj_brightness=1.0, adj_contrast=1.0):
    """ Evaluate model with dataloader

    Args:
        model (_type_): model weight
        dataloader (_type_): dataloader of [test or val]
        device (_type_): [gpu or cpu]
        criterion (_type_): loss module
        adj_brightness (float, optional): adjust brightness. Defaults to 1.0.
        adj_contrast (float, optional): adjust contrast. Defaults to 1.0.

    Returns:
        eval_loss, macro accuracy, micro accuracy, (precision/recall/f1-score) of (real class, fake class, micro average, macro average): metrics
    """
    loss = 0
    mac_accuracy = 0
    model.eval()
    # Find other metrics
    y_label = []
    y_pred_label = []
    begin = time.time()
    with torch.no_grad():
        for inputs0, fft_imgs0, labels0, inputs1, fft_imgs1, labels1, labels_contrastive in tqdm(dataloader):
            global_step += 1
            # Push to device
            inputs0, fft_imgs0, labels0 = inputs0.float().to(device), fft_imgs0.float().to(device), labels0.float().to(device)
            inputs1, fft_imgs1, labels1 = inputs1.float().to(device), fft_imgs1.float().to(device), labels1.float().to(device)
            labels_contrastive = labels_contrastive.float().to(device)

            # Forward
            embedding_0, logps0, embedding_1, logps1  = model.forward(inputs0, fft_imgs0, inputs1, fft_imgs1)     # Shape (32, 1)
            logps0 = logps0.squeeze()                     # Shape (32, )
            logps1 = logps1.squeeze()

            # Find mean loss
            bceloss_0 = bce_loss(logps0, labels0)
            bceloss_1 = bce_loss(logps1, labels1)
            contrastiveloss = contrastive_loss(embedding_0, embedding_1, labels_contrastive)
            batch_loss = bceloss_0 + bceloss_1 + contrastiveloss

            # Cumulate into running val loss
            loss += batch_loss.item()

            # Find accuracy
            equals = (labels0 == (logps0 > 0.5))
            mac_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps0.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)

    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros
    
def train_pairwise_dual_stream(model, margin=2, train_dir = '', val_dir ='', test_dir= '', image_size=128, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=30, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="pairwise-dual-efficient", args_txt=""):
    
    # Generate dataloader train and validation 
    dataloader_train, dataloader_val = generate_dataloader_dual_stream_for_pairwise(train_dir, val_dir, image_size, batch_size, num_workers)
    dataloader_test = generate_test_dataloader_dual_stream_for_pairwise(test_dir, image_size, 2*batch_size, num_workers)
    
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
                init_lr = lr * (0.8 ** ((init_epoch - 1) // 3))
                print('Resume epoch: {} - with step: {} - lr: {}'.format(init_epoch, init_step, init_lr))
            if 'step' in checkpoint:
                init_step = int(resume.split('_')[3])
                init_epoch = int(init_step / len(dataloader_train))
                init_lr = lr * (0.8 ** (init_epoch // 3))
                with open(osp.join(checkpoint, 'global_acc_loss.txt'), 'r') as f:
                    line = f.read().strip()
                    init_global_acc = float(line.split(',')[0])
                    init_global_loss = float(line.split(',')[1])
                print('Resume step: {} - in epoch: {} - lr: {} - global_acc: {} - global_loss: {}'.format(init_step, init_epoch, init_lr, init_global_acc, init_global_loss))              
        except:
            pass

    optimizer = define_optimizer(model, model_name=model_name, init_lr=init_lr, division_lr=division_lr, use_pretrained=use_pretrained)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3*i for i in range(1, epochs//3 + 1)], gamma = 0.8)
    
    # Define devices
    device = define_device(seed=seed, model_name=model_name)
        
    # Define criterion
    bce_loss = nn.BCELoss()
    contrastive_loss = ContrastiveLoss(device=device, margin=margin)
    bce_loss = bce_loss.to(device)
    contrastive_loss = contrastive_loss.to(device)
    
    # Define logging factor:
    ckc_pointdir, log, batch_writer, epoch_writer_tup, step_writer_tup = define_log_writer(checkpoint, resume, args_txt, (model, model_name, image_size))
    epoch_ckcpoint, epoch_val_writer, epoch_test_writer = epoch_writer_tup
    step_ckcpoint, step_val_writer, step_test_writer = step_writer_tup
        
    # Define Early stopping and Model saver
    early_stopping = EarlyStopping(patience=es_patience, verbose=True, tunning_metric=es_metric)
    epoch_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc', "test_realf1", "test_fakef1", "test_avgf1"])
    step_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc', "test_realf1", "test_fakef1", "test_avgf1"])
    
    # Define and load model
    model = model.to(device)
    if resume != "":
        model.load_state_dict(torch.load(osp.join(checkpoint, resume)))
    model.train()

    running_loss = 0
    running_acc = 0

    global_loss = init_global_loss
    global_acc = init_global_acc
    global_step = init_step
    
    for epoch in range(init_epoch, epochs):
        print("\n=========================================")
        print("Epoch: {}/{}".format(epoch+1, epochs))
        print("Model: {} - {}".format(model_name, args_txt))
        print("lr = ", [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))])

        # Train
        model.train()
        print("Training...")
        for inputs0, fft_imgs0, labels0, inputs1, fft_imgs1, labels1, labels_contrastive in tqdm(dataloader_train):
            global_step += 1
            # Push to device
            inputs0, fft_imgs0, labels0 = inputs0.float().to(device), fft_imgs0.float().to(device), labels0.float().to(device)
            inputs1, fft_imgs1, labels1 = inputs1.float().to(device), fft_imgs1.float().to(device), labels1.float().to(device)
            labels_contrastive = labels_contrastive.float().to(device)
            # Clear gradient after a step
            optimizer.zero_grad()

            # Forward netword
            embedding_0, logps0, embedding_1, logps1  = model.forward(inputs0, fft_imgs0, inputs1, fft_imgs1)     # Shape (32, 1)
            logps0 = logps0.squeeze()                     # Shape (32, )
            logps1 = logps1.squeeze()

            # Find mean loss
            bceloss_0 = bce_loss(logps0, labels0)
            bceloss_1 = bce_loss(logps1, labels1)
            contrastiveloss = contrastive_loss(embedding_0, embedding_1, labels_contrastive)
            loss = bceloss_0 + bceloss_1 + contrastiveloss
            
            # Backpropagation and update weights
            loss.backward()
            optimizer.step()

            # update running (train) loss and accuracy
            running_loss += loss.item()
            global_loss += loss.item()
            
            equals = (labels0 == (logps0 > 0.5))
            running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
            global_acc += torch.mean(equals.type(torch.FloatTensor)).item()

            # Save step's loss:
            # To tensorboard and to writer
            log.write_scalar(scalar_dict={"Loss/Single step": loss.item()}, global_step=global_step)
            batch_writer.write("{},{:.4f}\n".format(global_step, loss.item()))

            # Eval after <?> iters:
            stop = True
            if eval_per_iters != -1:
                if global_step % eval_per_iters == 0:
                    model.eval()
                    # Eval validation set
                    val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_pairwise_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                    # Eval test set
                    test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_pairwise_dual_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_test_writer, log, global_step, global_loss/global_step, global_acc/global_step, test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=False, phase="test")
                    # Save model:
                    step_model_saver(global_step, [val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2]], step_ckcpoint, model)
                    step_model_saver.save_last_model(step_ckcpoint, model, global_step)
                    step_model_saver.save_model(step_ckcpoint, model, global_step, save_ckcpoint=False, global_acc=global_acc, global_loss=global_loss)

                    es_cur_score = find_current_earlystopping_score(es_metric, val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2])
                    early_stopping(es_cur_score)
                    if early_stopping.early_stop:
                        print('Early stopping. Best {}: {:.6f}'.format(es_metric, early_stopping.best_score))
                        time.sleep(5)
                        os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
                        return
                    model.train()

        # Eval
        # print("Validating epoch...")
        # model.eval()
        # val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_pairwise_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        # save_result(epoch_val_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc/len(dataloader_train), val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=True, phase="val")
        # # Eval test set
        # test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_pairwise_dual_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        # save_result(epoch_test_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc/len(dataloader_train), test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=True, phase="test")
        # # Save model:
        # epoch_model_saver(epoch+1, [val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2]], epoch_ckcpoint, model)
        # epoch_model_saver.save_last_model(epoch_ckcpoint, model, epoch+1)
        # if refine_model(model_name=model_name):
        #     epoch_model_saver.save_model(epoch_ckcpoint, model, epoch+1)
        # Reset to the next epoch
        running_loss = 0
        running_acc = 0
        scheduler.step()
        model.train()

        ## Early stopping:
        # es_cur_score = find_current_earlystopping_score(es_metric, val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2])
        # early_stopping(es_cur_score)
        # if early_stopping.early_stop:
        #     print('Early stopping. Best {}: {:.6f}'.format(es_metric, early_stopping.best_score))
        #     break
    # Sleep 5 seconds for rename ckcpoint dir:
    time.sleep(5)
    os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    return