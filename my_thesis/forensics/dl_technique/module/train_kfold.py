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
from dataloader.KFold import CustomizeKFold
from loss.contrastive_loss import ContrastiveLoss

def define_log_writer_for_kfold(checkpoint: str, fold_idx: str, resume: str, args_txt:str, model: Tuple[torch.nn.Module, str, int]):
    """Defines some logging writer and saves model to text file

    Args:
        checkpoint (str): path to checkpoint directory
        args_txt (str): version of model
        model (Tuple[torch.nn.Module, str, int]): (model architecture, model name, image size)

    Returns:
        Tuple[]: (actual_checkpoint_dir, logger, writer for each batch  loss, List[epoch checkpoint dir, epoch writer for val, epoch writer for test], List[step checkpoint dir, step writer for val, step writer for test])
    """
    # Create checkpoint dir and sub-checkpoint dir for each hyperparameter:
    if not osp.exists(checkpoint):
        os.makedirs(checkpoint)
    ckc_pointdir = osp.join(checkpoint, args_txt + '/fold_{}'.format(fold_idx) if resume == '' else 'resume')
    if not osp.exists(ckc_pointdir):
        os.makedirs(ckc_pointdir)
    # Save log with tensorboard
    log = Logger(os.path.join(ckc_pointdir, "logs"))
    # Writer instance for <iter||loss per batch>
    batch_writer = open(osp.join(ckc_pointdir, 'batch loss.csv'), 'w')
    batch_writer.write("Iter, Loss per batch\n")
    
    ######################### Make directory for each type of evaluation #########################
    def make_sub_checkpoint(ckc_pointdir: str, eval_type="epoch", write_mode='w'):
        ckcpoint = osp.join(ckc_pointdir, eval_type)
        if not osp.exists(ckcpoint):
            os.mkdir(ckcpoint)
        # Writer instance for epoch validation set result
        val_writer = open(osp.join(ckcpoint, 'result_val.csv'), write_mode)
        val_header = "{}, Train loss, Train accuracy,".format(eval_type) +\
                    " Val loss, Val accuracy," +\
                    " Val real pre, Val real rec, Val real F1-Score," +\
                    " Val fake pre, Val fake rec, Val fake F1-Score," +\
                    " Val micro pre, Val micro rec, Val micro F1-Score," +\
                    " Val macro pre, Val macro rec, Val macro F1-Score\n"
        val_writer.write(val_header)
        # Writer instance for epoch validation test result
        test_writer = open(osp.join(ckcpoint, 'result_test.csv'), write_mode)
        test_header = "{}, Train loss, Train accuracy,".format(eval_type) +\
                    " Test loss, Test accuracy," +\
                    " Test real pre, Test real rec, Test real F1-Score," +\
                    " Test fake pre, Test fake rec, Test fake F1-Score," +\
                    " Test micro pre, Test micro rec, Val micro F1-Score," +\
                    " Test macro pre, Test macro rec, Test macro F1-Score\n"
        test_writer.write(test_header)
        return ckcpoint, val_writer, test_writer
        
    # Epoch and step save:
    epoch_ckcpoint, epoch_val_writer, epoch_test_writer = make_sub_checkpoint(ckc_pointdir, "epoch")
    step_ckcpoint, step_val_writer, step_test_writer = make_sub_checkpoint(ckc_pointdir, "step")

    # Save model to txt file
    sys.stdout = open(os.path.join(ckc_pointdir, 'model_{}.txt'.format(args_txt)), 'w')
    # if 'dual' in model[1]:
    #     if model[1] != 'pairwise_dual_efficient_vit' and model[1] != 'dual_cnn_feedforward_vit':
    #         torchsummary.summary(model[0], [(3, model[2], model[2]), (1, model[2], model[2])], device='cpu')
    # else:
    #     if model[1] != 'capsulenet':
    #         torchsummary.summary(model[0], (3, model[2], model[2]), device='cpu')
    # sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    return ckc_pointdir, log, batch_writer, (epoch_ckcpoint, epoch_val_writer, epoch_test_writer), (step_ckcpoint, step_val_writer, step_test_writer)

###################################################################
################# SINGLE CNN FOR RGB IMAGE STREAM #################
###################################################################

def eval_kfold_image_stream(model ,dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0 ):
    loss = 0
    mac_accuracy = 0
    model.eval()
    y_label = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Push to device
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            # Forward network
            logps = model.forward(inputs)
            if len(logps.shape) == 2:
                logps = logps.squeeze(dim=1)

            # Loss in a batch
            batch_loss = criterion(logps, labels)
            # Cumulate into running val loss
            loss += batch_loss.item()

            # Find accuracy
            equals = (labels == (logps > 0.5))
            mac_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            
    assert len(y_label) == len(y_pred_label), "Bug"
    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros

def train_kfold_image_stream(model_, what_fold='all', n_folds=5, use_trick=True, criterion_name=None, train_dir = '', val_dir ='', test_dir = '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
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
        epoch_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc'])
        step_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc'])
        
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
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                # Clear gradient after a step
                optimizer.zero_grad()

                # Forward netword
                logps = model(inputs)   # Shape (32, 1)
                logps = logps.squeeze()         # Shape (32, )

                # Find loss
                loss = criterion(logps, labels)
                # print("logps: ", logps, "     ====     loss: ", loss)
                
                # Backpropagation and update weights
                loss.backward()
                optimizer.step()

                # update running (train) loss and accuracy
                running_loss += loss.item()
                global_loss += loss.item()
                equals = (labels == (logps > 0.5))
                running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
                global_acc += torch.mean(equals.type(torch.FloatTensor)).item()

                # Save step's loss:
                # To tensorboard and to writer
                log.write_scalar(scalar_dict={"Loss/Single step": loss.item()}, global_step=global_step)
                batch_writer.write("{},{:.4f}\n".format(global_step, loss.item()))

                # Eval after <?> iters:
                if eval_per_iters != -1:
                    if (global_step % eval_per_iters == 0):
                        model.eval()
                        # Eval validation set
                        val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_kfold_image_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                        
                        # Eval test set
                        test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_kfold_image_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_test_writer, log, global_step, global_loss/global_step, global_acc/global_step, test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=False, phase="test")
                        # Save model:
                        step_model_saver(global_step, [val_loss, val_mic_acc, test_loss, test_mic_acc], step_ckcpoint, model)
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

def eval_kfold_capsulenet(capnet, vgg_ext, dataloader, device, capsule_loss, adj_brightness=1.0, adj_contrast=1.0 ):
    capnet.eval()
    vgg_ext.eval()

    y_label = []
    y_pred = []
    y_pred_label = []
    loss = 0
    mac_accuracy = 0
    
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
        mac_accuracy += metrics.accuracy_score(img_label, output_pred)
        
    mac_accuracy /= len(dataloader)
    loss /= len(dataloader)
    assert len(y_label) == len(y_pred_label), "Bug"
    ######## Calculate metrics:
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros

from model.cnn.capsule_net.model import VggExtractor, CapsuleNet
from loss.capsule_loss import CapsuleLoss
from torch.autograd import Variable
def train_kfold_capsulenet(what_fold='all', n_folds=5, use_trick=True, train_dir = '', val_dir ='', test_dir = '', gpu_id=0, beta1=0.9, dropout=0.05, image_size=128, lr=3e-4, \
              batch_size=16, num_workers=4, checkpoint='', resume='', epochs=20, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="capsule", args_txt="", dropout_in_mlp=True, augmentation=False):

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
    
        # Define devices
        device = define_device(seed=seed, model_name=model_name)
        
        # Define and load model
        vgg_ext = VggExtractor().to(device)
        capnet = CapsuleNet(num_class=2, device=device).to(device)
        
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
        
        optimizer = optim.Adam(capnet.parameters(), lr=init_lr, betas=(beta1, 0.999))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2*i for i in range(1, epochs//2 + 1)], gamma = 0.8)

        # Define criterion
        capsule_loss = CapsuleLoss().to(device)
        
        # Define logging factor:
        ckc_pointdir, log, batch_writer, epoch_writer_tup, step_writer_tup = define_log_writer_for_kfold(checkpoint, fold_idx, resume, args_txt, (capnet, model_name, image_size))
        epoch_ckcpoint, epoch_val_writer, epoch_test_writer = epoch_writer_tup
        step_ckcpoint, step_val_writer, step_test_writer = step_writer_tup
            
        # Define Early stopping and Model saver
        early_stopping = EarlyStopping(patience=es_patience, verbose=True, tunning_metric=es_metric)
        epoch_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc'])
        step_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc'])
        
        if resume != "":
            capnet.load_state_dict(torch.load(osp.join(checkpoint, resume)))
            capnet.train(mode=True)
            
            if device != 'cpu':
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

        global_loss = init_global_loss
        global_acc = init_global_acc
        global_step = init_step

        capnet.train()
        vgg_ext.train()

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

            y_label_step = []
            y_pred_label_step = []
            
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
                y_label_step.extend(img_label)
                y_pred_label_step.extend(output_pred)
                global_acc += metrics.accuracy_score(y_label_step, y_pred_label_step)

                # Save step's loss:
                # To tensorboard and to writer
                log.write_scalar(scalar_dict={"Loss/Single step": loss_dis_data}, global_step=global_step)
                batch_writer.write("{},{:.4f}\n".format(global_step, loss_dis_data))

                # Eval after <?> iters:
                if eval_per_iters != -1:
                    if global_step % eval_per_iters == 0:
                        capnet.eval()
                        # Eval validation set
                        val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_kfold_capsulenet(capnet, vgg_ext, dataloader_val, device, capsule_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                        # Eval test set
                        test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_kfold_capsulenet(capnet, vgg_ext, dataloader_test, device, capsule_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_test_writer, log, global_step, global_loss/global_step, global_acc/global_step, test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=False, phase="test")
                        # Save model:
                        step_model_saver(global_step, [val_loss, val_mic_acc, test_loss, test_mic_acc], step_ckcpoint, capnet)
                        step_model_saver.save_last_model(step_ckcpoint, capnet, global_step)
                        step_model_saver.save_model(step_ckcpoint, capnet, global_step, save_ckcpoint=False, global_acc=global_acc, global_loss=global_loss)
            
                        es_cur_score = find_current_earlystopping_score(es_metric, val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2])
                        early_stopping(es_cur_score)
                        if early_stopping.early_stop:
                            print('Early stopping. Best {}: {:.6f}'.format(es_metric, early_stopping.best_score))
                            time.sleep(5)
                            os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, args_txt if resume == '' else '', "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], 'fold' if resume == '' else 'resume', fold_idx)))
                            next_fold = True
                        if next_fold:
                            break
                        capnet.train()
                        vgg_ext.train()
                y_label_step = []
                y_pred_label_step = []

            running_acc = metrics.accuracy_score(y_label, y_pred_label)        

            if next_fold:
                break
        if next_fold:
            next_fold=False
            continue
  
            # Reset to the next epoch
            running_loss = 0
            running_acc = 0
            scheduler.step()
            capnet.train()
            # Early stopping:
            #
       # Sleep 5 seconds for rename ckcpoint dir:
        time.sleep(5)
        # Save epoch acc val, epoch acc test, step acc val, step acc test
        if osp.exists(ckc_pointdir):
            os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, args_txt if resume == '' else '', "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], 'fold' if resume == '' else 'resume', fold_idx)))
    return


############################################################################################################
################# DUAL CNN - <CNN/FEEDFORWARD> FOR RGB IMAGE AND FREQUENCY ANALYSIS STREAM #################
############################################################################################################
def eval_kfold_dual_stream(model, dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0):
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
        for inputs, fft_imgs, labels in dataloader:
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            # Push to device
            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), labels.float().to(device)

            # Forward network
            logps = model.forward(inputs, fft_imgs)

            if len(logps.shape) == 0:
                logps = logps.unsqueeze(dim=0)
            if len(logps.shape) == 2:
                logps = logps.squeeze(dim=1)

            # Loss in a batch
            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print("inputs: ", inputs)
            # print("logps shape: ", logps.shape)
            # print("labels shape: ", labels.shape)
            # print("logps: ", logps)
            # print("labels: ", labels)
            # sys.stdout = sys.__stdout__

            batch_loss = criterion(logps, labels)
            # Cumulate into running val loss
            loss += batch_loss.item()

            # Find accuracy
            equals = (labels == (logps > 0.5))
            mac_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)

    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros
    
def train_kfold_dual_stream(model_, what_fold='all', n_folds=5, use_trick=True, criterion_name=None, train_dir = '', val_dir ='', test_dir= '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=30, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="dual-efficient", args_txt="", augmentation=True, highpass=None):

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
        # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/inspect/61uadfv/train/fold_{}.txt'.format(fold_idx), 'w')
        # # print("\n=====================================================================================================")
        # # print("**** Train: ")
        # for img in trainset:
        #     print(img)
        # sys.stdout = sys.__stdout__
        # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/inspect/61uadfv/val/fold_{}.txt'.format(fold_idx), 'w')
        # # print("\n**** Val: ")
        # for img in valset:
        #     print(img)
        # sys.stdout = sys.__stdout__
        # continue

        dataloader_train, dataloader_val, num_samples = generate_dataloader_dual_cnn_stream_for_kfold(train_dir, trainset, valset, image_size, batch_size, num_workers, augmentation=augmentation, highpass=highpass)
        dataloader_test = generate_test_dataloader_dual_cnn_stream_for_kfold(test_dir, image_size, batch_size, num_workers, highpass=highpass)        
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
        epoch_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc'])
        step_model_saver = ModelSaver(save_metrics=["val_loss", "val_acc", "test_loss", 'test_acc'])
        
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
        cnt = 0
        stop_ = 0
        
        for epoch in range(init_epoch, epochs):
            print("\n=========================================")
            print("Epoch: {}/{}".format(epoch+1, epochs))
            print("Model: {} - {}".format(model_name, args_txt))
            print("lr = ", [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))])

            # Train
            model.train()
            print("Training...", len(dataloader_train))
            for inputs, fft_imgs, labels in tqdm(dataloader_train):
                global_step += 1
                # Push to device
                inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), labels.float().to(device)
                # Clear gradient after a step
                optimizer.zero_grad()

                # Forward netword
                logps = model.forward(inputs, fft_imgs)     # Shape (32, 1)
                logps = logps.squeeze()                     # Shape (32, )
                if len(logps.shape) == 0:
                    logps = logps.unsqueeze(dim=0)
                if len(logps.shape) == 2:
                    logps = logps.squeeze(dim=1)

                # Find mean loss
                loss = criterion(logps, labels)
                
                # Backpropagation and update weights
                loss.backward()
                optimizer.step()

                # update running (train) loss and accuracy
                running_loss += loss.item()
                global_loss += loss.item()
                
                equals = (labels == (logps > 0.5))
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
                        val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_kfold_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                        # Eval test set
                        test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_kfold_dual_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result(step_test_writer, log, global_step, global_loss/global_step, global_acc/global_step, test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=False, phase="test")
                        # Save model:
                        step_model_saver(global_step, [val_loss, val_mic_acc, test_loss, test_mic_acc], step_ckcpoint, model)
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

########################################################
################# PAIRWISE DUAL STREAM #################
########################################################
def define_log_writer_for_kfold_pairwise(checkpoint: str, fold_idx: int, resume: str, args_txt:str, model: Tuple[torch.nn.Module, str, int]):
    """Defines some logging writer and saves model to text file

    Args:
        checkpoint (str): path to checkpoint directory
        args_txt (str): version of model
        model (Tuple[torch.nn.Module, str, int]): (model architecture, model name, image size)

    Returns:
        Tuple[]: (actual_checkpoint_dir, logger, writer for each batch  loss, List[epoch checkpoint dir, epoch writer for val, epoch writer for test], List[step checkpoint dir, step writer for val, step writer for test])
    """
    # Create checkpoint dir and sub-checkpoint dir for each hyperparameter:
    if not osp.exists(checkpoint):
        os.makedirs(checkpoint)
    ckc_pointdir = osp.join(checkpoint, args_txt + '/fold_{}'.format(fold_idx) if resume == '' else 'resume')
    if not osp.exists(ckc_pointdir):
        os.makedirs(ckc_pointdir)
    # Save log with tensorboard
    log = Logger(os.path.join(ckc_pointdir, "logs"))
    # Writer instance for <iter||loss per batch>
    batch_contrast_writer = open(osp.join(ckc_pointdir, 'batch contrastive loss.csv'), 'w')
    batch_bce_writer = open(osp.join(ckc_pointdir, 'batch bce loss.csv'), 'w')
    batch_total_writer = open(osp.join(ckc_pointdir, 'batch total loss.csv'), 'w')
    batch_contrast_writer.write("Iter, Loss per batch\n")
    batch_bce_writer.write("Iter, Loss per batch\n")
    batch_total_writer.write("Iter, Loss per batch\n")
    
    ######################### Make directory for each type of evaluation #########################
    def make_sub_checkpoint(ckc_pointdir: str, eval_type="epoch", write_mode='w'):
        ckcpoint = osp.join(ckc_pointdir, eval_type)
        if not osp.exists(ckcpoint):
            os.mkdir(ckcpoint)
        # Writer instance for epoch validation set result
        val_writer = open(osp.join(ckcpoint, 'result_val.csv'), write_mode)
        val_header = "{}, Train contrastive loss, Train bce loss, Train total loss, Train accuracy,".format(eval_type) +\
                    " Val contrastive loss, Val bce loss, Val total loss, Val accuracy," +\
                    " Val real pre, Val real rec, Val real F1-Score," +\
                    " Val fake pre, Val fake rec, Val fake F1-Score," +\
                    " Val micro pre, Val micro rec, Val micro F1-Score," +\
                    " Val macro pre, Val macro rec, Val macro F1-Score\n"
        val_writer.write(val_header)
        # Writer instance for epoch validation test result
        test_writer = open(osp.join(ckcpoint, 'result_test.csv'), write_mode)
        test_header = "{}, Train contrastive loss, Train bce loss, Train total loss, Train accuracy,".format(eval_type) +\
                    " Test contrastive loss, Test bce loss, Test total loss, Test accuracy," +\
                    " Test real pre, Test real rec, Test real F1-Score," +\
                    " Test fake pre, Test fake rec, Test fake F1-Score," +\
                    " Test micro pre, Test micro rec, Val micro F1-Score," +\
                    " Test macro pre, Test macro rec, Test macro F1-Score\n"
        test_writer.write(test_header)
        return ckcpoint, val_writer, test_writer
        
    # Epoch and step save:
    epoch_ckcpoint, epoch_val_writer, epoch_test_writer = make_sub_checkpoint(ckc_pointdir, "epoch")
    step_ckcpoint, step_val_writer, step_test_writer = make_sub_checkpoint(ckc_pointdir, "step")    
    return ckc_pointdir, log, batch_contrast_writer, batch_bce_writer, batch_total_writer, (epoch_ckcpoint, epoch_val_writer, epoch_test_writer), (step_ckcpoint, step_val_writer, step_test_writer)

def save_result_for_pairwise(text_writer, log, iteration, train_contrastive_loss, train_bce_loss, train_total_loss, train_acc, val_contrastive_loss, val_bce_loss, val_total_loss, val_mac_acc, val_mic_acc, reals, fakes, micros, macros, is_epoch=True, phase='val'):
    #
    eval_type = "Epoch" if is_epoch else "Step"
    # print result
    result = "{}{} {} --- ".format('\n' if not is_epoch and phase == 'val' else '', eval_type, iteration)

    pre_real, rec_real, f1_real = reals
    pre_fake, rec_fake, f1_fake = fakes
    micro_pre, micro_rec, micro_f1 = micros
    macro_pre, macro_rec, macro_f1 = macros   
    
    # print(pre_real, rec_real, f1_real)
    # print(pre_fake, rec_fake, f1_fake)
    # print(micro_pre, micro_rec, micro_f1)

    result += "PHASE {} ##### Train c_loss: {:.4f}, b_loss: {:.4f}, t_loss: {:.4f} --- Train accuracy: {:.4f} --- ".format(phase, train_contrastive_loss, train_bce_loss, train_total_loss, train_acc)
    result += "{} c_loss: {:.4f}, b_loss: {:.4f}, t_loss: {:.4f} --- {} macro acc: {:.4f} --- {} micro acc: {:.4f} --- ".format(phase, val_contrastive_loss, val_bce_loss, val_total_loss, phase, val_mac_acc, phase, val_mic_acc)
    result += "{} real f1-score: {:.4f} --- ".format(phase, f1_real)
    result += "{} fake f1-score: {:.4f} --- ".format(phase, f1_fake)
    result += "{} avg f1-score: {:.4f} --- ".format(phase, macro_f1)
    print(result)
    
    # Save text result 
    save_txt  = "{},{:.6f},{:.6f},{:.6f},{:.6f},".format(iteration, train_contrastive_loss, train_bce_loss, train_total_loss, train_acc)
    save_txt += "{:.6f},{:.6f},{:.6f},{:.6f},".format(val_contrastive_loss, val_bce_loss, val_total_loss, val_mic_acc)
    save_txt += "{:.6f},{:.6f},{:.6f},".format(pre_real, rec_real, f1_real)
    save_txt += "{:.6f},{:.6f},{:.6f},".format(pre_fake, rec_fake, f1_fake)
    save_txt += "{:.6f},{:.6f},{:.6f},".format(micro_pre, micro_rec, micro_f1)
    save_txt += "{:.6f},{:.6f},{:.6f}\n".format(macro_pre, macro_rec, macro_f1)
    text_writer.write(save_txt)
    
    # Save log tensorboard
    scalar_dict_contrastive_loss = {
        "train contrastive loss": train_contrastive_loss, 
        "{} contrastive loss".format(phase): val_contrastive_loss
    }
    
    scalar_dict_bce_loss = {
        "train bce loss": train_bce_loss, 
        "{} bce loss".format(phase): val_bce_loss
    }

    scalar_dict_total_loss = {
        "train total loss": train_total_loss, 
        "{} total loss".format(phase): val_total_loss
    }

    scalar_dict_accuracy = {
        "train accuracy": train_acc, 
        "{} accuracy".format(phase): val_mic_acc
    }
    
    log.write_scalars(scalar_dict=scalar_dict_contrastive_loss, global_step=iteration, tag="Contrastive Loss "+eval_type)
    log.write_scalars(scalar_dict=scalar_dict_bce_loss, global_step=iteration, tag="BCE Loss "+eval_type)
    log.write_scalars(scalar_dict=scalar_dict_total_loss, global_step=iteration, tag="Total Loss "+eval_type)
    log.write_scalars(scalar_dict=scalar_dict_accuracy, global_step=iteration, tag="Accuracy "+eval_type)
    
    for cls, metrics in zip(['real', 'fake', 'micro', 'macro'], [reals, fakes, micros, macros]):
        pre, rec, f1 = metrics 
        scalar_dict_metric = {
            "precision": pre, 
            "recall": rec,
            "f1-score": f1
            
        }
        log.write_scalars(scalar_dict=scalar_dict_metric, global_step=iteration, tag="{} {} {}".format(phase, cls, eval_type))

def eval_kfold_pairwise_dual_stream(model, weight_importance, dataloader, device, bce_loss, contrastive_loss, adj_brightness=1.0, adj_contrast=1.0):
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
    contrastive_loss_ = 0
    bce_loss_ = 0
    total_loss_ = 0
    mac_accuracy = 0
    model.eval()
    # Find other metrics
    y_label = []
    y_pred_label = []
    begin = time.time()
    with torch.no_grad():
        for inputs0, fft_imgs0, labels0, inputs1, fft_imgs1, labels1, labels_contrastive in dataloader:
            # Push to device
            y_label.extend(labels0.cpu().numpy().astype(np.float64))
            inputs0, fft_imgs0, labels0 = inputs0.float().to(device), fft_imgs0.float().to(device), labels0.float().to(device)
            inputs1, fft_imgs1, labels1 = inputs1.float().to(device), fft_imgs1.float().to(device), labels1.float().to(device)
            labels_contrastive = labels_contrastive.float().to(device)
            
            # Forward
            embedding_0, logps0, embedding_1, logps1  = model.forward(inputs0, fft_imgs0, inputs1, fft_imgs1)     # Shape (32, 1)
            logps0 = logps0.squeeze()                     # Shape (32, )
            logps1 = logps1.squeeze()
            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print("logps0: ", logps0.shape, '          ===== ', logps0)
            # print("labels0: ", labels0.shape, '          ===== ', labels0)
            if len(logps0.shape) == 0:
                logps0 = logps0.unsqueeze(dim=0)
            if len(logps0.shape) == 2:
                logps0 = logps0.squeeze(dim=1)
            # Find mean loss

            # print("logps0: ", logps0.shape, '          ===== ', logps0)
            # print("labels0: ", labels0.shape, '          ===== ', labels0)
            sys.stdout = sys.__stdout__
            bceloss_0 = bce_loss(logps0, labels0)
            # bceloss_1 = bce_loss(logps1, labels1)
            contrastiveloss = contrastive_loss(embedding_0, embedding_1, labels_contrastive)
            batch_loss = weight_importance * bceloss_0 + contrastiveloss

            # Cumulate into running val loss
            contrastive_loss_ += contrastiveloss.item()
            bce_loss_ += bceloss_0.item()
            total_loss_ += batch_loss.item()

            # Find accuracy
            equals = (labels0 == (logps0 > 0.5))
            mac_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps0.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)

    ######## Calculate metrics:
    contrastive_loss_ /= len(dataloader)
    bce_loss_ /= len(dataloader)
    total_loss_ /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    # calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=True)
    return contrastive_loss_, bce_loss_, total_loss_, mac_accuracy, mic_accuracy, reals, fakes, micros, macros
    
def train_kfold_pairwise_dual_stream(model_, what_fold='all', n_folds=5, use_trick=True, weight_importance=2, margin=2, train_dir = '', val_dir ='', test_dir= '', image_size=128, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=30, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="pairwise_dual_cnn_vit", args_txt="", augmentation=True):

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
        # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/inspect/61uadfv/train/fold_{}.txt'.format(fold_idx), 'w')
        # # print("\n=====================================================================================================")
        # # print("**** Train: ")
        # for img in trainset:
        #     print(img)
        # sys.stdout = sys.__stdout__
        # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/inspect/61uadfv/val/fold_{}.txt'.format(fold_idx), 'w')
        # # print("\n**** Val: ")
        # for img in valset:
        #     print(img)
        # sys.stdout = sys.__stdout__
        # continue

        dataloader_train, dataloader_val, num_samples = generate_dataloader_dual_cnn_stream_for_kfold_pairwise(train_dir, trainset, valset, image_size, batch_size, num_workers, augmentation=augmentation)
        dataloader_test = generate_test_dataloader_dual_cnn_stream_for_kfold_pairwise(test_dir, image_size, batch_size, num_workers)    
    
        # Define optimizer (Adam) and learning rate decay
        init_lr = lr
        init_epoch = 0
        init_step = 0
        init_global_acc = 0
        init_global_bce_loss = 0
        init_global_contrastive_loss = 0
        init_global_total_loss = 0
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
                        init_global_bce_loss = float(line.split(',')[2])
                        init_global_contrastive_loss = float(line.split(',')[1])
                        init_global_total_loss = float(line.split(',')[3])

                    print('Resume step: {} - in epoch: {} - lr: {} - global_acc: {} - global_bce_loss: {} - global_contrastive_loss: {} - global_total_loss: {}'.format(init_step, init_epoch, init_lr, init_global_acc, init_global_bce_loss, init_global_contrastive_loss, init_global_total_loss))              
            except:
                pass

        model = copy.deepcopy(model_copy)
        optimizer = define_optimizer(model, model_name=model_name, init_lr=init_lr, division_lr=division_lr, use_pretrained=use_pretrained)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3*i for i in range(1, epochs//3 + 1)], gamma = 0.8)
        
        # Define devices
        device = define_device(seed=seed, model_name=model_name)
            
        # Define criterion
        bce_loss = nn.BCELoss()
        # bce_loss = WBCE(weights=[1.0, 1.0])
        contrastive_loss = ContrastiveLoss(device=device, margin=margin)
        bce_loss = bce_loss.to(device)
        contrastive_loss = contrastive_loss.to(device)
        
        # Define logging factor:
        ckc_pointdir, log, batch_contrast_writer, batch_bce_writer, batch_total_writer, epoch_writer_tup, step_writer_tup = define_log_writer_for_kfold_pairwise(checkpoint, fold_idx, resume, args_txt, (model, model_name, image_size))
        epoch_ckcpoint, epoch_val_writer, epoch_test_writer = epoch_writer_tup
        step_ckcpoint, step_val_writer, step_test_writer = step_writer_tup
            
        # Define Early stopping and Model saver
        early_stopping = EarlyStopping(patience=es_patience, verbose=True, tunning_metric=es_metric)
        epoch_model_saver = ModelSaver(save_metrics=["val_bceloss", "val_totalloss", "val_acc", "test_bceloss", "test_totalloss", 'test_acc'])
        step_model_saver = ModelSaver(save_metrics=["val_bceloss", "val_totalloss", "val_acc", "test_bceloss", "test_totalloss", 'test_acc'])
        
        # Define and load model
        model = model.to(device)
        if resume != "":
            model.load_state_dict(torch.load(osp.join(checkpoint, resume)))
        model.train()

        running_loss = 0
        running_acc = 0

        global_contrastive_loss = init_global_contrastive_loss
        global_bce_loss = init_global_bce_loss
        global_total_loss = init_global_total_loss
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
                # bceloss_1 = bce_loss(logps1, labels1)
                contrastiveloss = contrastive_loss(embedding_0, embedding_1, labels_contrastive)
                loss = weight_importance * bceloss_0 + contrastiveloss

                if global_step % 500 == 0:
                    print("Bceloss 0: {}  --- Contrastive: {} - Total: {} ".format(bceloss_0.item(), contrastiveloss.item(), loss.item()))
                
                # Backpropagation and update weights
                loss.backward()
                optimizer.step()

                # update running (train) loss and accuracy
                running_loss += loss.item()
                global_contrastive_loss += contrastiveloss.item()
                global_bce_loss += bceloss_0.item()
                global_total_loss += loss.item()
                
                equals = (labels0 == (logps0 > 0.5))
                running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
                global_acc += torch.mean(equals.type(torch.FloatTensor)).item()

                # Save step's loss:
                # To tensorboard and to writer
                log.write_scalar(scalar_dict={"Contrastive Loss/Single Step": contrastiveloss.item()}, global_step=global_step)
                log.write_scalar(scalar_dict={"BCE Loss/Single Step": bceloss_0.item()}, global_step=global_step)
                log.write_scalar(scalar_dict={"Total Loss/Single Step": loss.item()}, global_step=global_step)
                batch_contrast_writer.write("{},{:.4f}\n".format(global_step, contrastiveloss.item()))
                batch_bce_writer.write("{},{:.4f}\n".format(global_step, bceloss_0.item()))
                batch_total_writer.write("{},{:.4f}\n".format(global_step, loss.item()))

                # Eval after <?> iters:
                stop = True
                if eval_per_iters != -1:
                    if global_step % eval_per_iters == 0:
                        model.eval()
                        # Eval validation set
                        val_contrastive_loss, val_bce_loss, val_total_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_kfold_pairwise_dual_stream(model, weight_importance, dataloader_val, device, bce_loss, contrastive_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result_for_pairwise(step_val_writer, log, global_step, global_contrastive_loss/global_step, global_bce_loss/global_step, global_total_loss/global_step, global_acc/global_step, val_contrastive_loss, val_bce_loss, val_total_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                        # Eval test set
                        test_contrastive_loss, test_bce_loss, test_total_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_kfold_pairwise_dual_stream(model, weight_importance, dataloader_test, device, bce_loss, contrastive_loss, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                        save_result_for_pairwise(step_test_writer, log, global_step, global_contrastive_loss/global_step, global_bce_loss/global_step, global_total_loss/global_step, global_acc/global_step, test_contrastive_loss, test_bce_loss, test_total_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=False, phase="test")
                        # Save model:
                        step_model_saver(global_step, [val_bce_loss, val_total_loss, val_mic_acc, test_bce_loss, test_total_loss, test_mic_acc], step_ckcpoint, model)
                        step_model_saver.save_last_model(step_ckcpoint, model, global_step)
                        step_model_saver.save_model_for_pairwise(step_ckcpoint, model, global_step, save_ckcpoint=False, global_acc=global_acc, global_contrastive_loss=global_contrastive_loss, global_bce_loss=global_bce_loss, global_total_loss=global_total_loss)

                        es_cur_score = find_current_earlystopping_score(es_metric, val_total_loss, val_mic_acc, test_total_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2])
                        early_stopping(es_cur_score)

                        if early_stopping.early_stop:
                            print('Early stopping. Best {}: {:.6f}'.format(es_metric, early_stopping.best_score))
                            time.sleep(5)
                            os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, args_txt if resume == '' else '', "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[2], step_model_saver.best_scores[3], step_model_saver.best_scores[5], 'fold' if resume == '' else 'resume', fold_idx)))
                            next_fold = True
                        if next_fold:
                            break
                        model.train()
            if next_fold:
                break
        if next_fold:
            next_fold=False
            continue

            running_loss = 0
            running_acc = 0
            scheduler.step()
            model.train()

        # Sleep 5 seconds for rename ckcpoint dir:
        time.sleep(5)
        # Save epoch acc val, epoch acc test, step acc val, step acc test
        if osp.exists(ckc_pointdir):
            os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, args_txt if resume == '' else '', "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[2], step_model_saver.best_scores[3], step_model_saver.best_scores[5], 'fold' if resume == '' else 'resume', fold_idx)))
    return