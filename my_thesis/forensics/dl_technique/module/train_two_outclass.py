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
from torch.autograd import Variable

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
# warnings.filterwarnings("default")
from sklearn.exceptions import UndefinedMetricWarning

from model.cnn.capsule_net.model import VggExtractor, CapsuleNet
from loss.capsule_loss import CapsuleLoss
from module.train_torch import define_log_writer, define_device, calculate_metric, save_result, find_current_earlystopping_score
from metrics.metric import calculate_cls_metrics

def eval_capsulenet(capnet, vgg_ext, dataloader, device, capsule_loss, adj_brightness=1.0, adj_contrast=1.0 ):
    capnet.eval()

    y_label = []
    y_pred = []
    y_pred_label = []
    loss = 0
    
    for inputs, labels in dataloader:
        labels[labels > 1] = 1
        img_label = labels.numpy().astype(np.float)
        inputs, labels = inputs.to(device), labels.to(device)

        input_v = Variable(inputs)
        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=False)

        loss_dis = capsule_loss(classes, Variable(labels, requires_grad=False))
        loss_dis_data = loss_dis.item()
        output_dis = class_.data.cpu().numpy()

        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        loss += loss_dis_data
        y_label.extend(img_label)
        y_pred.extend(output_dis)
        y_pred_label.extend(output_pred)
        
    mac_accuracy = 0
    loss /= len(dataloader)
    assert len(y_label) == len(y_pred_label), "Bug"
    ######## Calculate metrics:
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros

def train_capsulenet(train_dir = '', val_dir ='', test_dir = '', gpu_id=0, beta1=0.9, dropout=0.05, image_size=128, lr=3e-4, \
              batch_size=16, num_workers=4, checkpoint='', resume='', epochs=20, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="xception", args_txt=""):
    # Generate dataloader train and validation 
    dataloader_train, dataloader_val, num_samples = generate_dataloader_image_stream(train_dir, val_dir, image_size, batch_size, num_workers)
    dataloader_test = generate_test_dataloader_image_stream(test_dir, image_size, batch_size, num_workers)
    
    # Define devices
    device = define_device(seed=seed)
    
    # Define and load model
    vgg_ext = VggExtractor().to(device)
    capnet = CapsuleNet(num_class=2, device=device).to(device)
    
    # Define optimizer (Adam) and learning rate decay
    init_lr = lr
    init_epoch = 0
    init_step = 0
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
                print('Resume step: {} - in epoch: {} - lr: {}'.format(init_step, init_epoch, init_lr))              
        except:
            pass
        
    optimizer = optim.Adam(capnet.parameters(), lr=init_lr, betas=(beta1, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3*i for i in range(1, epochs//3 + 1)], gamma = 0.8)

    # Define criterion
    capsule_loss = CapsuleLoss().to(device)
    
    # Define logging factor:
    ckc_pointdir, log, batch_writer, epoch_writer_tup, step_writer_tup = define_log_writer(checkpoint, resume, args_txt, (capnet, model_name, image_size))
    epoch_ckcpoint, epoch_val_writer, epoch_test_writer = epoch_writer_tup
    step_ckcpoint, step_val_writer, step_test_writer = step_writer_tup
        
    # Define Early stopping and Model saver
    early_stopping = EarlyStopping(patience=es_patience, verbose=True, tunning_metric=es_metric)
    epoch_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc', "test_realf1", "test_fakef1", "test_avgf1"])
    step_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc', "test_realf1", "test_fakef1", "test_avgf1"])
    
    if resume != "":
        capnet.load_state_dict(torch.load(osp.join(checkpoint, resume)))
        capnet.train(mode=True)
        
        if device != 'cpu':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    global_loss = 0.0
    global_step = init_step
    capnet.train()

    for epoch in range(init_epoch, epochs):
        print("\n=========================================")
        print("Epoch: {}/{}".format(epoch+1, epochs))
        print("Model: {} - {}".format(model_name, args_txt))
        print("lr = ", optimizer.param_groups[0]['lr'])

        # Train
        capnet.train()
        running_loss = 0
        running_acc = 0
        y_label = np.array([], dtype=np.float)
        y_pred_label = np.array([], dtype=np.float)
        
        print("Training...")
        for inputs, labels in tqdm(dataloader_train):
            global_step += 1
            # Push to device
            labels[labels > 1] = 1
            img_label = labels.numpy().astype(np.float)
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear gradient after a step
            optimizer.zero_grad()

            # Forward network
            input_v = Variable(inputs)
            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=True, dropout=dropout)

            # Find loss
            loss_dis = capsule_loss(classes, Variable(labels, requires_grad=False))
            loss_dis_data = loss_dis.item()
            
            # Backpropagation and update weights
            loss_dis.backward()
            optimizer.step()

            # update running (train) loss and accuracy
            output_dis = class_.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0
                    
            y_label = np.concatenate((y_label, img_label))
            y_pred_label = np.concatenate((y_pred_label, output_pred))
                    
            running_loss += loss_dis_data
            global_loss += loss_dis_data

            # Save step's loss:
            # To tensorboard and to writer
            log.write_scalar(scalar_dict={"Loss/Single step": loss_dis_data}, global_step=global_step)
            batch_writer.write("{},{:.4f}\n".format(global_step, loss_dis_data))

            # Eval after <?> iters:
            if eval_per_iters != -1:
                if global_step % eval_per_iters == 0:
                    capnet.eval()
                    # Eval validation set
                    val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_capsulenet(capnet, vgg_ext, dataloader_val, device, capsule_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_val_writer, log, global_step, global_loss/global_step, 0, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                    # Eval test set
                    test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_capsulenet(capnet, vgg_ext, dataloader_test, device, capsule_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_test_writer, log, global_step, global_loss/global_step, 0, test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=False, phase="test")
                    # Save model:
                    step_model_saver(global_step, [val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2]], step_ckcpoint, capnet)
                    step_model_saver.save_last_model(step_ckcpoint, capnet, global_step)
                    capnet.train()
                    
        running_acc = metrics.accuracy_score(y_label, y_pred_label)
        
        # Eval
        print("Validating epoch...")
        capnet.eval()
        val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_capsulenet(capnet, vgg_ext, dataloader_val, device, capsule_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        save_result(epoch_val_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=True, phase="val")
        # Eval test set
        test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_capsulenet(capnet, vgg_ext, dataloader_test, device, capsule_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        save_result(epoch_test_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc, test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=True, phase="test")
        # Save model:
        epoch_model_saver(epoch+1, [val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2]], epoch_ckcpoint, capnet)
        epoch_model_saver.save_last_model(epoch_ckcpoint, capnet, epoch+1)
        
        # Reset to the next epoch
        running_loss = 0
        running_acc = 0
        scheduler.step()
        capnet.train()

        # Early stopping:
        es_cur_score = find_current_earlystopping_score(es_metric, val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2])
        early_stopping(es_cur_score)
        if early_stopping.early_stop:
            print('Early stopping. Best {}: {:.6f}'.format(es_metric, early_stopping.best_score))
            break
    time.sleep(5)
    os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(epoch_model_saver.best_scores[3], step_model_saver.best_scores[3], epoch_model_saver.best_scores[2], step_model_saver.best_scores[2], args_txt if resume == '' else 'resume')))
    return