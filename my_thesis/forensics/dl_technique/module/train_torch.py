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

def refine_model(model_name: str):
    if 'dual' in model_name and 'vit' in model_name:
        return True
    return False

def find_current_earlystopping_score(es_metric, val_loss, val_mic_acc, test_loss, test_mic_acc, test_real_f1, test_fake_f1, test_macro_f1):
    if es_metric == 'val_loss':
        return val_loss
    if es_metric == 'val_acc':
        return val_mic_acc
    if es_metric == 'test_loss':
        return test_loss
    if es_metric == 'test_acc':
        return test_mic_acc
    if es_metric == 'test_realf1':
        return test_real_f1
    if es_metric == 'test_fakef1':
        return test_fake_f1
    if es_metric == 'test_avgf1':
        return test_macro_f1
    
def save_result(text_writer, log, iteration, train_loss, train_acc, val_loss, val_mac_acc, val_mic_acc, reals, fakes, micros, macros, is_epoch=True, phase='val'):
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

    result += "PHASE {} ##### Train loss: {:.4f} --- Train accuracy: {:.4f} --- ".format(phase, train_loss, train_acc)
    result += "{} loss: {:.4f} --- {} macro acc: {:.4f} --- {} micro acc: {:.4f} --- ".format(phase, val_loss, phase, val_mac_acc, phase, val_mic_acc)
    result += "{} real f1-score: {:.4f} --- ".format(phase, f1_real)
    result += "{} fake f1-score: {:.4f} --- ".format(phase, f1_fake)
    result += "{} avg f1-score: {:.4f} --- ".format(phase, macro_f1)
    print(result)
    
    # Save text result 
    save_txt  = "{},{:.6f},{:.6f},".format(iteration, train_loss, train_acc)
    save_txt += "{:.6f},{:.6f},".format(val_loss, val_mic_acc)
    save_txt += "{:.6f},{:.6f},{:.6f},".format(pre_real, rec_real, f1_real)
    save_txt += "{:.6f},{:.6f},{:.6f},".format(pre_fake, rec_fake, f1_fake)
    save_txt += "{:.6f},{:.6f},{:.6f},".format(micro_pre, micro_rec, micro_f1)
    save_txt += "{:.6f},{:.6f},{:.6f}\n".format(macro_pre, macro_rec, macro_f1)
    text_writer.write(save_txt)
    
    # Save log tensorboard
    scalar_dict_loss = {
        "train loss": train_loss, 
        "{} loss".format(phase): val_loss
    }
    
    scalar_dict_accuracy = {
        "train accuracy": train_acc, 
        "{} accuracy".format(phase): val_mic_acc
    }
    
    log.write_scalars(scalar_dict=scalar_dict_loss, global_step=iteration, tag="Loss "+eval_type)
    log.write_scalars(scalar_dict=scalar_dict_accuracy, global_step=iteration, tag="Accuracy "+eval_type)
    
    for cls, metrics in zip(['real', 'fake', 'micro', 'macro'], [reals, fakes, micros, macros]):
        pre, rec, f1 = metrics 
        scalar_dict_metric = {
            "precision": pre, 
            "recall": rec,
            "f1-score": f1
            
        }
        log.write_scalars(scalar_dict=scalar_dict_metric, global_step=iteration, tag="{} {} {}".format(phase, cls, eval_type))

def define_criterion(criterion_name: List[str], num_samples: int):
    criterion = None
    if criterion_name[0] == 'bce':
        criterion = nn.BCELoss()
        print("Use binary cross entropy loss.")
    elif criterion_name[0] == 'ce':
        criterion = nn.CrossEntropyLoss()
        print("Use cross entropy loss.")
    else:
        if criterion_name[0] == 'cbce':
            criterion = WBCE(weights=[1.0, 1.0])
            print("Use custom binary cross entropy loss.")
        if criterion_name[0] == 'wbce':
            weight_class_0 = 0.5 * (num_samples[0] + num_samples[1]) / (num_samples[0])
            weight_class_1 = 0.5 * (num_samples[0] + num_samples[1]) / (num_samples[1])
            weights = [weight_class_0, weight_class_1]
            criterion = WBCE(weights=weights)
            print("Use weighted cross entropy loss: ", weights)
        elif criterion_name[0] == 'focalloss':
            weight_class_0 = num_samples[1] / (num_samples[0] + num_samples[1])
            weight_class_1 = 1 - weight_class_0
            # weights = [weight_class_0, weight_class_1]
            weights = [1.0, 1.0]
            criterion = FL(weights=weights, gamma=float(criterion_name[1]))
            print("Use focal loss: ", weights, "gamma: ", float(criterion_name[1]))
    return criterion

def define_log_writer(checkpoint: str, resume: str, args_txt:str, model: Tuple[torch.nn.Module, str, int]):
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
    ckc_pointdir = osp.join(checkpoint, args_txt if resume == '' else 'resume')
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

def define_device(seed: int, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
    #     # if 'dual' in model_name and 'vit' in model_name:
    #     #     return device
        cudnn.deterministic = True
    return device

def define_optimizer(model, model_name: str, init_lr: float, division_lr=True, use_pretrained=False):
    if not division_lr or not use_pretrained:
        optimizer = optim.Adam(model.parameters(), lr=init_lr)
    else:
        extractor = []
        rest = []
        for name, param in model.named_parameters():
            if 'extractor' in name:
                extractor.append(param)
            else:
                rest.append(param)
        optimizer = Adam(
            [
                {"params": extractor, "lr": init_lr/5},
                {"params": rest, "lr": init_lr},
            ],
        )
    return optimizer


def calculate_metric(y_label: List[float], y_pred_label: List[float]):
    mic_accuracy = accuracy_score(y_label, y_pred_label)
    try:
        precision_fake = precision_score(y_label, y_pred_label)
        recall_fake = recall_score(y_label, y_pred_label)
        f1_fake = f1_score(y_label, y_pred_label)
    except UndefinedMetricWarning:
        print("* Fake: ")
        print("* y_pred_label: {}\n".format(len(y_pred_label)), y_pred_label)
        print("* y_label: {}\n".format(len(y_label)), y_label)
        exit(0)

    try:
        precision_real = precision_score(y_label, y_pred_label, pos_label=0)
        recall_real = recall_score(y_label, y_pred_label, pos_label=0)
        f1_real = f1_score(y_label, y_pred_label, pos_label=0)
    except UndefinedMetricWarning:
        print("* Real: ")
        print("* y_pred_label: {}\n".format(len(y_pred_label)), y_pred_label)
        print("* y_label: {}\n".format(len(y_label)), y_label)
        exit(0)

    macro_precision = precision_score(y_label, y_pred_label, average='macro')
    macro_recall = recall_score(y_label, y_pred_label, average='macro')
    macro_f1 = f1_score(y_label, y_pred_label, average='macro')
    
    micro_precision = precision_score(y_label, y_pred_label, average='micro')
    micro_recall = recall_score(y_label, y_pred_label, average='micro')
    micro_f1 = f1_score(y_label, y_pred_label, average='micro')
    return mic_accuracy, (precision_real, recall_real, f1_real),\
                         (precision_fake, recall_fake, f1_fake),\
                         (micro_precision, micro_recall, micro_f1),\
                         (macro_precision, macro_recall, macro_f1)
    
###################################################################
################# SINGLE CNN FOR RGB IMAGE STREAM #################
###################################################################

def eval_image_stream(model ,dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0 ):
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

def train_image_stream(model, criterion_name=None, train_dir = '', val_dir ='', test_dir = '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=20, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="xception", args_txt="", augmentation=True, gpu_id=2):
    # Generate dataloader train and validation 
    dataloader_train, dataloader_val, num_samples = generate_dataloader_single_cnn_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=augmentation)
    dataloader_test = generate_test_dataloader_single_cnn_stream(test_dir, image_size, batch_size, num_workers)
    
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
        
    optimizer = define_optimizer(model, model_name=model_name, init_lr=init_lr, division_lr=division_lr, use_pretrained=use_pretrained)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2*i for i in range(1, epochs//2 + 1)], gamma = 0.8)
    
    # Define devices
    device = define_device(seed=seed, model_name=model_name)

    # Define criterion
    criterion = define_criterion(criterion_name, num_samples)
    criterion = criterion.to(device)
    
    # Define logging factor:
    ckc_pointdir, log, batch_writer, epoch_writer_tup, step_writer_tup = define_log_writer(checkpoint, resume, args_txt, (model, model_name, image_size))
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
                    val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_image_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                    
                    # Eval test set
                    test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_image_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
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
                        os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
                        return
                    model.train()

        # Eval
        # print("Validating epoch...")
        # model.eval()
        # val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_image_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        # save_result(epoch_val_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc/len(dataloader_train), val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=True, phase="val")
        # # Eval test set
        # test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_image_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        # save_result(epoch_test_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc/len(dataloader_train), test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros, is_epoch=True, phase="test")
        # # Save model:
        # epoch_model_saver(epoch+1, [val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2]], epoch_ckcpoint, model)
        # epoch_model_saver.save_last_model(epoch_ckcpoint, model, epoch+1)
        
        # Reset to the next epoch
        running_loss = 0
        running_acc = 0
        scheduler.step()
        model.train()

        # # Early stopping:
        # es_cur_score = find_current_earlystopping_score(es_metric, val_loss, val_mic_acc, test_loss, test_mic_acc, test_reals[2], test_fakes[2], test_macros[2])
        # early_stopping(es_cur_score)
        # if early_stopping.early_stop:
        #     print('Early stopping. Best {}: {:.6f}'.format(es_metric, early_stopping.best_score))
        #     break
    time.sleep(5)
    # Save epoch acc val, epoch acc test, step acc val, step acc test
    os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    # os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(epoch_model_saver.best_scores[1], epoch_model_saver.best_scores[3], step_model_saver.best_scores[1], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    return

############################################################################################################
################# DUAL CNN - <CNN/FEEDFORWARD> FOR RGB IMAGE AND FREQUENCY ANALYSIS STREAM #################
############################################################################################################
def eval_dual_stream(model, dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0):
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
            if len(logps.shape) == 2:
                logps = logps.squeeze(dim=1)

            # Loss in a batch
            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print("inputs shape, fft images shape: ", inputs.shape, fft_imgs.shape)
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
    
def train_dual_stream(model, criterion_name=None, train_dir = '', val_dir ='', test_dir= '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=30, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="dual-efficient", args_txt="", augmentation=True):
    
    # Generate dataloader train and validation 
    if model_name != "dual_cnn_feedforward_vit":
        dataloader_train, dataloader_val, num_samples = generate_dataloader_dual_cnn_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=augmentation)
        dataloader_test = generate_test_dataloader_dual_cnn_stream(test_dir, image_size, 2*batch_size, num_workers)
    else:
        dataloader_train, dataloader_val, num_samples = generate_dataloader_dual_cnnfeedforward_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=augmentation)
        dataloader_test = generate_test_dataloader_dual_cnnfeedforward_stream(test_dir, image_size, 2*batch_size, num_workers)        
    
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

    optimizer = define_optimizer(model, model_name=model_name, init_lr=init_lr, division_lr=division_lr, use_pretrained=use_pretrained)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2*i for i in range(1, epochs//2 + 1)], gamma = 0.8)
    
    # Define devices
    device = define_device(seed=seed, model_name=model_name)
        
    # Define criterion
    criterion = define_criterion(criterion_name, num_samples)
    criterion = criterion.to(device)
    
    # Define logging factor:
    ckc_pointdir, log, batch_writer, epoch_writer_tup, step_writer_tup = define_log_writer(checkpoint, resume, args_txt, (model, model_name, image_size))
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
        print("Training...")
        for inputs, fft_imgs, labels in tqdm(dataloader_train):
            global_step += 1
            # Push to device
            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), labels.float().to(device)
            # Clear gradient after a step
            optimizer.zero_grad()

            # Forward netword
            logps = model.forward(inputs, fft_imgs)     # Shape (32, 1)
            logps = logps.squeeze()                     # Shape (32, )

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
                    val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                    # Eval test set
                    test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_dual_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
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
                        os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
                        return
                    model.train()

        # Eval
        # print("Validating epoch...")
        # model.eval()
        # val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        # save_result(epoch_val_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc/len(dataloader_train), val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=True, phase="val")
        # # Eval test set
        # test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_dual_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
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
    # Save epoch acc val, epoch acc test, step acc val, step acc test
    os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    # os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(epoch_model_saver.best_scores[1], epoch_model_saver.best_scores[3], step_model_saver.best_scores[1], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    return

############################################################################################################
################# TRIPLE CNN - <CNN/FEEDFORWARD> FOR RGB IMAGE AND FREQUENCY ANALYSIS STREAM #################
############################################################################################################
def eval_triple_stream(model, dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0):
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
        for inputs, mag_imgs, phase_imgs, labels in dataloader:
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            # Push to device
            inputs, mag_imgs, phase_imgs, labels = inputs.float().to(device), mag_imgs.float().to(device), phase_imgs.float().to(device), labels.float().to(device)

            # Forward network
            logps = model.forward(inputs, mag_imgs, phase_imgs)
            if len(logps.shape) == 0:
                logps = logps.unsqueeze(dim=0)
            if len(logps.shape) == 2:
                logps = logps.squeeze(dim=1)

            # Loss in a batch
            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print("inputs shape, fft images shape: ", inputs.shape, fft_imgs.shape)
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
    
def train_triple_stream(model, criterion_name=None, train_dir = '', val_dir ='', test_dir= '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=30, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="dual-efficient", args_txt="", augmentation=True):
    
    # Generate dataloader train and validation 
    dataloader_train, dataloader_val, num_samples = generate_dataloader_triple_cnn_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=augmentation)
    dataloader_test = generate_test_dataloader_triple_cnn_stream(test_dir, image_size, 2*batch_size, num_workers)

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

    optimizer = define_optimizer(model, model_name=model_name, init_lr=init_lr, division_lr=division_lr, use_pretrained=use_pretrained)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2*i for i in range(1, epochs//2 + 1)], gamma = 0.8)
    
    # Define devices
    device = define_device(seed=seed, model_name=model_name)
        
    # Define criterion
    criterion = define_criterion(criterion_name, num_samples)
    criterion = criterion.to(device)
    
    # Define logging factor:
    ckc_pointdir, log, batch_writer, epoch_writer_tup, step_writer_tup = define_log_writer(checkpoint, resume, args_txt, (model, model_name, image_size))
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
        print("Training...")
        for inputs, mag_imgs, phase_imgs, labels in tqdm(dataloader_train):
            global_step += 1
            # Push to device
            inputs, mag_imgs, phase_imgs, labels = inputs.float().to(device), mag_imgs.float().to(device), phase_imgs.float().to(device), labels.float().to(device)
            # Clear gradient after a step
            optimizer.zero_grad()

            # Forward netword
            logps = model.forward(inputs, mag_imgs, phase_imgs)     # Shape (32, 1)
            logps = logps.squeeze()                                 # Shape (32, )

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
                    val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_triple_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                    # Eval test set
                    test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_triple_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
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
                        os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
                        return
                    model.train()

        # Eval
        # print("Validating epoch...")
        # model.eval()
        # val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        # save_result(epoch_val_writer, log, epoch+1, running_loss/len(dataloader_train), running_acc/len(dataloader_train), val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=True, phase="val")
        # # Eval test set
        # test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_dual_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
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
    # Save epoch acc val, epoch acc test, step acc val, step acc test
    os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    # os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(epoch_model_saver.best_scores[1], epoch_model_saver.best_scores[3], step_model_saver.best_scores[1], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    return