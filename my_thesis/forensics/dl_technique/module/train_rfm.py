import torch
import os, sys
import os.path as osp
import time
import numpy as np
from tqdm import tqdm
import random

from sklearn.metrics import recall_score, accuracy_score, precision_score, log_loss, classification_report, f1_score
from dataloader.gen_dataloader import *
from module.train_torch import define_device, define_optimizer, define_log_writer, save_result, calculate_metric, calculate_cls_metrics, find_current_earlystopping_score
from module.utils import *
from utils.EarlyStopping import EarlyStopping
from utils.ModelSaver import ModelSaver
from sklearn import metrics
from torch import optim
import torch.nn as nn


###################################################################
################# SINGLE CNN FOR RGB IMAGE STREAM #################
###################################################################

def eval_rfm(model ,dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0 ):
    loss = 0
    mac_accuracy = 0
    model.eval()
    y_label = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Push to device
            img_label = labels.cpu().numpy().astype(np.float64)
            # labels_cpu = labels.cpu().numpy()
            # new_labels = []
            # for label in labels_cpu:
            #     if label[0] == 0:
            #         new_labels.append([1, 0])
            #     else:
            #         new_labels.append([0, 1])
            # labels = torch.tensor(new_labels)
            inputs, labels = inputs.float().to(device), torch.tensor(labels, dtype=torch.int64).to(device)

            # Forward network
            outputs = model.forward(inputs)

            # Loss in a batch
            batch_loss = criterion(outputs, labels)
            # Cumulate into running val loss
            loss += batch_loss.item()

            output_cpu = outputs.data.cpu().numpy()
            output_pred = np.zeros((output_cpu.shape[0]), dtype=np.float)

            for i in range(output_cpu.shape[0]):
                if output_cpu[i,1] >= output_cpu[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            y_pred_label.extend(output_pred)
            y_label.extend(img_label)
            mac_accuracy += metrics.accuracy_score(img_label, output_pred)
            
    assert len(y_label) == len(y_pred_label), "Bug"
    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros

def train_rfm(model, train_dir = '', val_dir ='', test_dir = '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=20, eval_per_iters=-1, seed=0, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="xception", args_txt="", augmentation=True):
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
    criterion = nn.CrossEntropyLoss()
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

    for epoch in range(init_epoch, epochs):
        print("\n=========================================")
        print("Epoch: {}/{}".format(epoch+1, epochs))
        print("Model: {} - {}".format(model_name, args_txt))
        print("lr = ", optimizer.param_groups[0]['lr'])

        # Train
        model.train()
        y_label = []
        y_pred_label = []
        print("Training...")
        for inputs, labels in tqdm(dataloader_train):
            global_step += 1
            img_label = labels.numpy().astype(np.float)

            ''' ↓ the implementation of RFM ↓ '''
            model.eval()
            inputs = inputs.to(device)
            mask = cal_fam(model, inputs)
            imgmask = torch.ones_like(mask)
            imgh = imgw = image_size

            for i in range(len(mask)):
                maxind = np.argsort(mask[i].cpu().numpy().flatten())[::-1]
                pointcnt = 0
                for pointind in maxind:
                    pointx = pointind // imgw
                    pointy = pointind % imgw

                    if imgmask[i][0][pointx][pointy] == 1:

                        maskh = random.randint(1, image_size//2)
                        maskw = random.randint(1, image_size//2)

                        sh = random.randint(1, maskh)
                        sw = random.randint(1, maskw)

                        top = max(pointx-sh, 0)
                        bot = min(pointx+(maskh-sh), imgh)
                        lef = max(pointy-sw, 0)
                        rig = min(pointy+(maskw-sw), imgw)

                        imgmask[i][:, top:bot, lef:rig] = torch.zeros_like(imgmask[i][:, top:bot, lef:rig])

                        pointcnt += 1
                        if pointcnt >= 3:
                            break

            inputs = imgmask * inputs + (1-imgmask) * (torch.rand_like(inputs)*2-1.)
            ''' ↑ the implementation of RFM ↑ '''
            
            # labels_cpu = labels.cpu().numpy()
            # new_labels = []
            # for label in labels_cpu:
            #     if label == 0:
            #         new_labels.append([1, 0])
            #     else:
            #         new_labels.append([0, 1])
            # labels = torch.tensor(new_labels, dtype=torch.int64)
            inputs, labels = inputs.float().to(device), torch.tensor(labels, dtype=torch.int64).to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)   # Shape (32, 1)

            # Find loss
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            
            # Backpropagation and update weights
            loss.backward()
            optimizer.step()

            # update running (train) loss and accuracy
            running_loss += loss.item()
            global_loss += loss.item()

            output_cpu = outputs.data.cpu().numpy()
            output_pred = np.zeros((output_cpu.shape[0]), dtype=np.float)
            for i in range(output_cpu.shape[0]):
                if output_cpu[i,1] >= output_cpu[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            y_label.extend(img_label)
            y_pred_label.extend(output_pred)
            acc = metrics.accuracy_score(y_label, y_pred_label)
            y_label = []
            y_pred_label = []

            running_acc += acc
            global_acc += acc

            # Save step's loss:
            # To tensorboard and to writer
            log.write_scalar(scalar_dict={"Loss/Single step": loss.item()}, global_step=global_step)
            batch_writer.write("{},{:.4f}\n".format(global_step, loss.item()))

            # Eval after <?> iters:
            if eval_per_iters != -1:
                if (global_step % eval_per_iters == 0):
                    model.eval()
                    # Eval validation set
                    val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_rfm(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                    
                    # Eval test set
                    test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_rfm(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
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

        running_loss = 0
        running_acc = 0
        scheduler.step()
        model.train()

    time.sleep(5)
    # Save epoch acc val, epoch acc test, step acc val, step acc test
    os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    # os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(epoch_model_saver.best_scores[1], epoch_model_saver.best_scores[3], step_model_saver.best_scores[1], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    return

############################################################################################################
################# DUAL CNN - <CNN/FEEDFORWARD> FOR RGB IMAGE AND FREQUENCY ANALYSIS STREAM #################
############################################################################################################
def eval_rfm_dual_stream(model, dataloader, device, criterion, adj_brightness=1.0, adj_contrast=1.0):
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
            img_label = labels.cpu().numpy().astype(np.float64)
            # Push to device
            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), torch.tensor(labels, dtype=torch.int64).to(device)

            # Forward network
            outputs = model.forward(inputs, fft_imgs)

            # Loss in a batch
            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print("inputs shape, fft images shape: ", inputs.shape, fft_imgs.shape)
            # print("logps shape: ", logps.shape)
            # print("labels shape: ", labels.shape)
            # print("logps: ", logps)
            # print("labels: ", labels)
            # sys.stdout = sys.__stdout__

            batch_loss = criterion(outputs, labels)
            # Cumulate into running val loss
            loss += batch_loss.item()

            # Find accuracy
            output_cpu = outputs.data.cpu().numpy()
            output_pred = np.zeros((output_cpu.shape[0]), dtype=np.float)

            for i in range(output_cpu.shape[0]):
                if output_cpu[i,1] >= output_cpu[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            y_pred_label.extend(output_pred)
            y_label.extend(img_label)
            mac_accuracy += metrics.accuracy_score(img_label, output_pred)

    ######## Calculate metrics:
    loss /= len(dataloader)
    mac_accuracy /= len(dataloader)
    # built-in methods for calculating metrics
    mic_accuracy, reals, fakes, micros, macros = calculate_metric(y_label, y_pred_label)
    calculate_cls_metrics(y_label=np.array(y_label, dtype=np.float64), y_pred_label=np.array(y_pred_label, dtype=np.float64), save=True, print_metric=False)
    return loss, mac_accuracy, mic_accuracy, reals, fakes, micros, macros
    
def train_rfm_dual_stream(model, train_dir = '', val_dir ='', test_dir= '', image_size=256, lr=3e-4, division_lr=True, use_pretrained=False,\
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
    criterion = nn.CrossEntropyLoss()
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
    
    for epoch in range(init_epoch, epochs):
        print("\n=========================================")
        print("Epoch: {}/{}".format(epoch+1, epochs))
        print("Model: {} - {}".format(model_name, args_txt))
        print("lr = ", [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))])

        # Train
        model.train()
        print("Training...")
        y_label_step = []
        y_pred_label_step = []
        for inputs, fft_imgs, labels in tqdm(dataloader_train):
            global_step += 1
            img_label = labels.numpy().astype(np.float)

            ''' ↓ the implementation of RFM ↓ '''
            model.eval()
            inputs, fft_imgs = inputs.to(device), fft_imgs.to(device)
            mask = cal_fam(model, inputs, fft_imgs)
            imgmask = torch.ones_like(mask)
            imgh = imgw = image_size

            for i in range(len(mask)):
                maxind = np.argsort(mask[i].cpu().numpy().flatten())[::-1]
                pointcnt = 0
                for pointind in maxind:
                    pointx = pointind // imgw
                    pointy = pointind % imgw

                    if imgmask[i][0][pointx][pointy] == 1:

                        maskh = random.randint(1, image_size//2)
                        maskw = random.randint(1, image_size//2)

                        sh = random.randint(1, maskh)
                        sw = random.randint(1, maskw)

                        top = max(pointx-sh, 0)
                        bot = min(pointx+(maskh-sh), imgh)
                        lef = max(pointy-sw, 0)
                        rig = min(pointy+(maskw-sw), imgw)

                        imgmask[i][:, top:bot, lef:rig] = torch.zeros_like(imgmask[i][:, top:bot, lef:rig])

                        pointcnt += 1
                        if pointcnt >= 3:
                            break
            inputs = imgmask * inputs + (1-imgmask) * (torch.rand_like(inputs)*2-1.)
            ''' ↑ the implementation of RFM ↑ '''

            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), torch.tensor(labels, dtype=torch.int64).to(device)
            # Clear gradient after a step
            optimizer.zero_grad()

            # Forward netword
            outputs = model.forward(inputs, fft_imgs)     # Shape (32, 1)
            # logps = logps.squeeze()                     # Shape (32, )

            # Find mean loss
            loss = criterion(outputs, labels)
            
            # Backpropagation and update weights
            loss.backward()
            optimizer.step()

            # update running (train) loss and accuracy
            running_loss += loss.item()
            global_loss += loss.item()
            
            output_cpu = outputs.data.cpu().numpy()
            output_pred = np.zeros((output_cpu.shape[0]), dtype=np.float)
            for i in range(output_cpu.shape[0]):
                if output_cpu[i,1] >= output_cpu[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            y_label.extend(img_label)
            y_pred_label.extend(output_pred)
            acc = metrics.accuracy_score(y_label, y_pred_label)
            y_label = []
            y_pred_label = []

            running_acc += acc
            global_acc += acc

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
                    val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros = eval_rfm_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    save_result(step_val_writer, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mac_acc, val_mic_acc, val_reals, val_fakes, val_micros, val_macros, is_epoch=False, phase="val")
                    # Eval test set
                    test_loss, test_mac_acc, test_mic_acc, test_reals, test_fakes, test_micros, test_macros = eval_rfm_dual_stream(model, dataloader_test, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
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

        running_loss = 0
        running_acc = 0
        scheduler.step()
        model.train()

    time.sleep(5)
    # Save epoch acc val, epoch acc test, step acc val, step acc test
    os.rename(src=ckc_pointdir, dst=osp.join(checkpoint, "({:.4f}_{:.4f}_{:.4f}_{:.4f})_{}".format(step_model_saver.best_scores[0], step_model_saver.best_scores[1], step_model_saver.best_scores[2], step_model_saver.best_scores[3], args_txt if resume == '' else 'resume')))
    return