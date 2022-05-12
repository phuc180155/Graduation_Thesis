import torch
import os, sys
import os.path as osp
import time
import numpy as np
from tqdm import tqdm

from sklearn.metrics import recall_score, accuracy_score, precision_score, log_loss, classification_report, f1_score
from dataloader.gen_dataloader import *

####################################################
########### SINGLE SPATIAL IMAGE STREAM ############
####################################################

def eval_image_stream(model, criterion, test_dir="", image_size=256,\
                      batch_size=16, num_workers=8, checkpoint="checkpoint/...", resume="",\
                      adj_brightness=1.0, adj_contrast=1.0, show_time=False, model_name="", args_txt=""):
    # Define devive and push model, loss module into device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Load trained model
    try:
        model.load_state_dict(torch.load(osp.join(checkpoint, resume)))
    except:
        print("ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("model khong ton tai : ", osp.join(checkpoint, resume) + '.pt')
        return

    # Generate test dataloader
    dataloader_test = generate_test_dataloader_image_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=adj_brightness,adj_contrast=adj_contrast)

    test_loss = 0.0           # global loss of all images
    test_accuracy = 0.0       # global accuracy of all images
    model.eval()
    
    y_label = []              # save ground truth labels  [0, 0, 1, 1...]           - (num_images, ) 
    y_pred = []               # save output of all images [0.05, 0.4, 0.6, 0.4...]  - (num_images, )
    y_pred_label = []         # save predicted labels     [0, 0, 1, 0...]           - (num_images, )
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_test):
            begin = time.time()
            # Push groundtruth label (original) to cpu and save in a list
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            
            # Push input to device and forward to model
            inputs, labels = inputs.to(device), labels.float().to(device)
            logps = model.forward(inputs)   # (batch_size, 1)
            logps = logps.squeeze()         # (batch_size, )

            # Push predicted output - in [0, 1] to cpu and save in a list
            logps_cpu = logps.cpu().numpy()
            y_pred.extend(logps_cpu.astype(np.float64))
            
            # Show batch processing time
            if show_time:
                print("Time : ", time.time() - begin)
                
            # Calculate loss
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Find predicted label - is 0 or 1 and save in list
            equals = labels == (logps > 0.5)
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            
            # Calculate mean accuracy
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    ######## Calculate metrics:
    test_loss /= len(dataloader_test)
    test_accuracy /= len(dataloader_test)
    # built-in methods for calculating metrics
    logloss = log_loss(y_label, y_pred, labels=np.array([0., 1.]))
    accuracy = accuracy_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    f1 = 2.0 * recall * precision / (recall + precision)
    
    print("Test loss: {:.3f}.. \n".format(test_loss) +\
          "Test accuracy: {:.3f}\n".format(test_accuracy) +\
          "Test log_loss: {:.3f}\n".format(logloss) +\
          "Test accuracy_score: {:.3f}\n".format(accuracy) +\
          "Test precision_score: {:.3f}\n".format(precision) +\
          "Test recall: {:.3f}\n".format(recall) +\
          "Test F1: {:.3f}".format(f1))
    print(classification_report(y_label,y_pred_label))
    return


##########################################################
########### DUAL SPATIAL AND FREQUENCY STREAM ############
##########################################################
def eval_dual_stream(model, criterion, test_dir='',image_size=256, \
                 batch_size=16, num_workers=8, checkpoint="checkpoint", resume="", \
                 adj_brightness=1.0, adj_contrast=1.0, show_time=False, model_name="", args_txt=""):

    # Defines device and push model and loss module to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    # Load trained model
    try:
        model.load_state_dict(torch.load(osp.join(checkpoint, resume)))
    except:
        print("ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("model khong ton tai : ", osp.join(checkpoint, resume) + '.pt')
        return

    # Make dataloader
    dataloader_test = generate_test_dataloader_dual_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=adj_brightness, adj_contrast=adj_contrast)

    test_loss = 0.0           # global loss of all images
    test_accuracy = 0.0       # global accuracy of all images
    model.eval()
    
    y_label = []              # save ground truth labels  [0, 0, 1, 1...]           - (num_images, ) 
    y_pred = []               # save output of all images [0.05, 0.4, 0.6, 0.4...]  - (num_images, )
    y_pred_label = []         # save predicted labels     [0, 0, 1, 0...]           - (num_images, )
    
    with torch.no_grad():
        for inputs, fft_imgs, labels in tqdm(dataloader_test):
            begin = time.time()
            # Push groundtruth label (original) to cpu and save in a list
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            # Push input to device and forward to model
            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), labels.float().to(device)
            logps = model.forward(inputs, fft_imgs)      # shape (batchsize, 1)
            logps = logps.squeeze()                      # shape (batchsize, )   
            
            # Push predicted output - in [0, 1] to cpu and save in a list
            logps_cpu = logps.cpu().numpy()
            y_pred.extend(logps_cpu.astype(np.float64))
            
            # Show batch processing time
            if show_time:
                print("Time : ", time.time() - begin)
                
            # Calculate loss
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Find predicted label, 0 or 1 and save in a list
            equals = labels == (logps > 0.5)
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            
            # Calculate mean accuracy
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    ######## Calculate metrics:
    test_loss /= len(dataloader_test)
    test_accuracy /= len(dataloader_test)
    # built-in methods for calculating metrics
    logloss = log_loss(y_label, y_pred, labels=np.array([0., 1.]))
    accuracy = accuracy_score(y_label, y_pred_label)
    
    precision_fake = precision_score(y_label, y_pred_label)
    recall_fake = recall_score(y_label, y_pred_label)
    f1_fake = f1_score(y_label, y_pred_label)
    
    precision_real = precision_score(y_label, y_pred_label, pos_label=0)
    recall_real = recall_score(y_label, y_pred_label, pos_label=0)
    f1_real = f1_score(y_label, y_pred_label, pos_label=0)

    macro_precision = precision_score(y_label, y_pred_label, average='macro')
    macro_recall = recall_score(y_label, y_pred_label, average='macro')
    macro_f1 = f1_score(y_label, y_pred_label, average='macro')
    
    micro_precision = precision_score(y_label, y_pred_label, average='micro')
    micro_recall = recall_score(y_label, y_pred_label, average='micro')
    micro_f1 = f1_score(y_label, y_pred_label, average='micro')
    
    print("Test loss: {:.3f}.. \n".format(test_loss) +\
          "Test accuracy: {:.3f}\n".format(test_accuracy) +\
          "Test log_loss: {:.3f}\n".format(logloss) +\
          "Test accuracy_score: {:.3f}\n".format(accuracy))
    
    print("Test precision class real: {:.3f}\n".format(precision_real) +\
          "Test recall class real: {:.3f}\n".format(recall_real) +\
          "Test F1 class real: {:.3f}".format(f1_real))
    
    print("Test precision class fake: {:.3f}\n".format(precision_fake) +\
          "Test recall class fake: {:.3f}\n".format(recall_fake) +\
          "Test F1 class fake: {:.3f}".format(f1_fake))
    
    print("Test micro precision: {:.3f}\n".format(micro_precision) +\
          "Test micro recall: {:.3f}\n".format(micro_recall) +\
          "Test micro F1: {:.3f}".format(micro_f1))
    
    print("Test macro precision: {:.3f}\n".format(macro_precision) +\
          "Test macro recall: {:.3f}\n".format(macro_recall) +\
          "Test macro F1: {:.3f}".format(macro_f1))
    
    print(classification_report(y_label,y_pred_label))
    # save:
    # epoch, val_loss, acc, pre_0, rec_0, f1_0, pre_1, rec_1, f1_1, mic_pre, mic_rec, mic_f1, mac_pre, mac_rec, mac_f1
    with open(osp.join(checkpoint, 'result.csv'), 'a') as f:
        epoch = int(resume.replace('.pt', '').split('_')[1])
        f.write('{},{:.4},{:.4},{:4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n'.format(epoch, test_loss, accuracy,\
                                                                                                                precision_real, recall_real, f1_real,\
                                                                                                                precision_fake, recall_fake, f1_fake,\
                                                                                                                micro_precision, micro_recall, micro_f1,\
                                                                                macro_precision, macro_recall, macro_f1))

    return