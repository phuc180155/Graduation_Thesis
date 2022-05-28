import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import torch
import torch.nn as nn
import argparse
import json
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_dir', type=str, default="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/train", help="path to train data")
    parser.add_argument('--val_dir', type=str, default="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/val", help="path to validation data")
    parser.add_argument('--test_dir', type=str, default="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/test", help="path to test data")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=2, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = "checkpoint/datasetv5/dfdcv5/dual_patch_cnn_cma_vit", help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 2, help='GPU id ')
    parser.add_argument('--resume',type=str, default = '', help='Resume from checkpoint ')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loss',type=str, default = "bce", help='Loss function use')
    parser.add_argument('--gamma',type=float, default=0.0, help="gamma hyperparameter for focal loss")
    parser.add_argument('--eval_per_iters',type=int, default=1000, help='Evaluate per some iterations')
    parser.add_argument('--es_metric',type=str, default='none', help='Criterion for early stopping')
    parser.add_argument('--es_patience',type=int, default=5, help='Early stopping epoch')
    # Add for avoiding overfitting
    parser.add_argument('--dropout_in_mlp', type=float, required=True, default=0.2)
    parser.add_argument('--augmentation', type=int, required=True, default=0)
    
    sub_parser = parser.add_subparsers(dest="model", help="Choose model of the available ones:  xception, efficient_dual, ViT, CrossViT, efficient ViT, dual efficient vit...")
    
    ######################## CNN architecture ########################
    parser_capsule = sub_parser.add_parser('capsule', help='CapsuleNet')
    parser_capsule.add_argument("--beta",type=int,required=False,default=0.9,help="Beta for optimizer Adam")
    parser_capsule.add_argument("--dropout", type=float, required=False, default=0.05)
    
    parser_xception = sub_parser.add_parser('xception', help='XceptionNet')
    parser_xception.add_argument('--pretrained', type=int, default=0, required=True)
    parser_xception_rfm = sub_parser.add_parser('xception_rfm', help='XceptionNet')
    parser_xception_rfm.add_argument('--pretrained', type=int, default=0, required=True)
    parser_meso4 = sub_parser.add_parser('meso4', help='MesoNet')

    parser_dual_eff = sub_parser.add_parser('dual_efficient', help="Efficient-Frequency Net")
    parser_dual_eff.add_argument('--pretrained', type=int, default=0, required=True)
    parser_srm_2_stream = sub_parser.add_parser('srm_two_stream', help="SRM 2 stream net from \"Generalizing Face Forgery Detection with High-frequency Features (CVPR 2021).\"")
    
    # Ablation study 1: Remove ViT
    parser_dual_attn_cnn = sub_parser.add_parser('dual_attn_cnn', help="Ablation Study")
    parser_dual_attn_cnn.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_attn_cnn.add_argument("--version",type=str, default="ca-fadd-0.8", required=False, help="Some changes in model")
    parser_dual_attn_cnn.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_attn_cnn.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_attn_cnn.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_attn_cnn.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_dual_attn_cnn.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_attn_cnn.add_argument("--conv_attn", type=int, default=0, help="")   
    parser_dual_attn_cnn.add_argument("--ratio", type=int, default=1, help="")   
    parser_dual_attn_cnn.add_argument("--qkv_embed", type=int, default=1, help="")   
    parser_dual_attn_cnn.add_argument("--inner_ca_dim", type=int, default=0, help="") 
    parser_dual_attn_cnn.add_argument("--init_ca_weight", type=int, default=1, help="") 
    parser_dual_attn_cnn.add_argument("--prj_out", type=int, default=0, help="")
    parser_dual_attn_cnn.add_argument("--act", type=str, default='relu', help="")
    parser_dual_attn_cnn.add_argument("--init_weight", type=int, default=0, help="")
    parser_dual_attn_cnn.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_dual_attn_cnn.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_dual_attn_cnn.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_dual_attn_cnn.add_argument("--division_lr", type=int, default=0, help="")

    parser_dual_cma_cnn_attn = sub_parser.add_parser('dual_cma_cnn_attn', help="Ablation Study")
    parser_dual_cma_cnn_attn.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_cma_cnn_attn.add_argument("--version",type=str, default="ca-fadd-0.8", required=False, help="Some changes in model")
    parser_dual_cma_cnn_attn.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_cma_cnn_attn.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_cma_cnn_attn.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_cma_cnn_attn.add_argument("--gamma_cma", type=float, default=0.5)
    parser_dual_cma_cnn_attn.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_dual_cma_cnn_attn.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_cma_cnn_attn.add_argument("--conv_attn", type=int, default=0, help="")   
    parser_dual_cma_cnn_attn.add_argument("--ratio", type=int, default=1, help="")   
    parser_dual_cma_cnn_attn.add_argument("--qkv_embed", type=int, default=1, help="")   
    parser_dual_cma_cnn_attn.add_argument("--inner_ca_dim", type=int, default=0, help="") 
    parser_dual_cma_cnn_attn.add_argument("--init_ca_weight", type=int, default=1, help="") 
    parser_dual_cma_cnn_attn.add_argument("--prj_out", type=int, default=0, help="")
    parser_dual_cma_cnn_attn.add_argument("--act_fusion", type=str, default='relu', help="")
    parser_dual_cma_cnn_attn.add_argument("--act_cma", type=str, default='relu', help="")
    parser_dual_cma_cnn_attn.add_argument("--init_weight", type=int, default=0, help="")
    parser_dual_cma_cnn_attn.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_dual_cma_cnn_attn.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_dual_cma_cnn_attn.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_dual_cma_cnn_attn.add_argument("--division_lr", type=int, default=0, help="")
    
    ######################## Vision transformer architecture ########################
    parser.add_argument('--dim',type=int, default = 1024, help='dim of embeding')
    parser.add_argument('--depth',type=int, default = 6, help='Number of attention layer in transformer module')
    parser.add_argument('--heads',type=int, default = 8, help='number of head in attention layer')
    parser.add_argument('--mlp_dim',type=int, default = 2048, help='dim of hidden layer in transformer layer')
    parser.add_argument('--dim_head',type=int, default = 64, help='in transformer layer ')
    parser.add_argument('--pool',type=str, default = "cls", help='in transformer layer ')
    
    parser_vit = sub_parser.add_parser('vit', help='ViT transformer Net')
    parser_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")

    parser_efficientvit = sub_parser.add_parser('efficient_vit', help='CrossViT transformer Net')
    parser_efficientvit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_efficientvit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_efficientvit.add_argument("--freeze", type=int, default=0, help="Freeze backbone")
    parser_efficientvit.add_argument("--division_lr", type=int, default=0, help="")

    parser_swin_vit = sub_parser.add_parser('swin_vit', help='Swin transformer')
    parser_swin_vit.add_argument("--window_size",type=int,default=7,help="window size in swin vit")

    parser_m2tr = sub_parser.add_parser('m2tr', help='')
    parser_m2tr.add_argument("--backbone",type=str,default="efficientnet-b0",help="")
    parser_m2tr.add_argument("--texture_layer",type=str,default="",help="")
    parser_m2tr.add_argument("--feature_layer",type=str,default="",help="")
    parser_m2tr.add_argument("--depth",type=int,default=0,help="")
    parser_m2tr.add_argument("--drop_ratio",type=float,default=0.0,help="")
    parser_m2tr.add_argument("--has_decoder",type=int,default=0,help="")
    
    # My refined model:
    parser_dual_cnn_vit = sub_parser.add_parser('dual_cnn_vit', help='My model')
    parser_dual_cnn_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_cnn_vit.add_argument("--version",type=str, default="ca-fadd-0.8", required=False, help="Some changes in model")
    parser_dual_cnn_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_cnn_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_cnn_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_cnn_vit.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_dual_cnn_vit.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_cnn_vit.add_argument("--conv_attn", type=int, default=0, help="")   
    parser_dual_cnn_vit.add_argument("--ratio", type=int, default=1, help="")   
    parser_dual_cnn_vit.add_argument("--qkv_embed", type=int, default=1, help="")   
    parser_dual_cnn_vit.add_argument("--inner_ca_dim", type=int, default=0, help="") 
    parser_dual_cnn_vit.add_argument("--init_ca_weight", type=int, default=1, help="") 
    parser_dual_cnn_vit.add_argument("--prj_out", type=int, default=0, help="")
    parser_dual_cnn_vit.add_argument("--act", type=str, default='relu', help="")
    parser_dual_cnn_vit.add_argument("--position_embed", type=int, default=1, help="")
    parser_dual_cnn_vit.add_argument("--init_weight", type=int, default=0, help="")
    parser_dual_cnn_vit.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_dual_cnn_vit.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_dual_cnn_vit.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_dual_cnn_vit.add_argument("--division_lr", type=int, default=0, help="")
    parser_dual_cnn_vit.add_argument("--classifier", type=str, default="mlp", help="")
    ########### TEST ###########
    parser_dual_cnn_vit_test = sub_parser.add_parser('dual_cnn_vit_test', help='My model')
    parser_dual_cnn_vit_test.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_cnn_vit_test.add_argument("--version",type=str, default="ca-fadd-0.8", required=False, help="Some changes in model")
    parser_dual_cnn_vit_test.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_cnn_vit_test.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_cnn_vit_test.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_cnn_vit_test.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_dual_cnn_vit_test.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_cnn_vit_test.add_argument("--conv_attn", type=int, default=0, help="")   
    parser_dual_cnn_vit_test.add_argument("--ratio", type=int, default=1, help="")   
    parser_dual_cnn_vit_test.add_argument("--qkv_embed", type=int, default=1, help="")   
    parser_dual_cnn_vit_test.add_argument("--inner_ca_dim", type=int, default=0, help="") 
    parser_dual_cnn_vit_test.add_argument("--init_ca_weight", type=int, default=1, help="") 
    parser_dual_cnn_vit_test.add_argument("--prj_out", type=int, default=0, help="")
    parser_dual_cnn_vit_test.add_argument("--act", type=str, default='relu', help="")
    parser_dual_cnn_vit_test.add_argument("--position_embed", type=int, default=1, help="")
    parser_dual_cnn_vit_test.add_argument("--init_weight", type=int, default=0, help="")
    parser_dual_cnn_vit_test.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_dual_cnn_vit_test.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_dual_cnn_vit_test.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_dual_cnn_vit_test.add_argument("--division_lr", type=int, default=0, help="")
    parser_dual_cnn_vit_test.add_argument("--classifier", type=str, default="", help="")

    parser_dual_cma_cnn_vit = sub_parser.add_parser('dual_cma_cnn_vit', help='My model')
    parser_dual_cma_cnn_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_cma_cnn_vit.add_argument("--version",type=str, default="ca-fadd-0.8", required=False, help="Some changes in model")
    parser_dual_cma_cnn_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_cma_cnn_vit.add_argument("--gamma_cma", type=float, default=0.5)
    parser_dual_cma_cnn_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_cma_cnn_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_cma_cnn_vit.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_dual_cma_cnn_vit.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_cma_cnn_vit.add_argument("--conv_attn", type=int, default=0, help="")   
    parser_dual_cma_cnn_vit.add_argument("--ratio", type=int, default=1, help="")   
    parser_dual_cma_cnn_vit.add_argument("--qkv_embed", type=int, default=1, help="")   
    parser_dual_cma_cnn_vit.add_argument("--inner_ca_dim", type=int, default=0, help="") 
    parser_dual_cma_cnn_vit.add_argument("--init_ca_weight", type=int, default=1, help="") 
    parser_dual_cma_cnn_vit.add_argument("--prj_out", type=int, default=0, help="")
    parser_dual_cma_cnn_vit.add_argument("--act_fusion", type=str, default='relu', help="")
    parser_dual_cma_cnn_vit.add_argument("--act_cma", type=str, default='relu', help="")
    parser_dual_cma_cnn_vit.add_argument("--position_embed", type=int, default=1, help="")
    parser_dual_cma_cnn_vit.add_argument("--init_weight", type=int, default=0, help="")
    parser_dual_cma_cnn_vit.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_dual_cma_cnn_vit.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_dual_cma_cnn_vit.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_dual_cma_cnn_vit.add_argument("--division_lr", type=int, default=0, help="")

    parser_dual_cnn_cma_transformer = sub_parser.add_parser('dual_cnn_cma_transformer', help='My model')
    parser_dual_cnn_cma_transformer.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_cnn_cma_transformer.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_cnn_cma_transformer.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_cnn_cma_transformer.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_dual_cnn_cma_transformer.add_argument("--act", type=str, default='relu', help="")
    parser_dual_cnn_cma_transformer.add_argument("--position_embed", type=int, default=1, help="")
    parser_dual_cnn_cma_transformer.add_argument("--init_type", type=str, default="normal", help="")
    parser_dual_cnn_cma_transformer.add_argument("--division_lr", type=int, default=0, help="")
    parser_dual_cnn_cma_transformer.add_argument("--depth", type=int, default=4, help="")
    parser_dual_cnn_cma_transformer.add_argument("--patch_resolution", type=str, default="1-2-4-8", help="")

    parser_dual_patch_cnn_cma_vit = sub_parser.add_parser('dual_patch_cnn_cma_vit', help='My model') 
    parser_dual_patch_cnn_cma_vit.add_argument("--flatten_type",type=str, default="patch", required=False, help="Type of backbone")
    parser_dual_patch_cnn_cma_vit.add_argument("--patch_size",type=int, default=2, required=False, help="Type of backbone")
    parser_dual_patch_cnn_cma_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_patch_cnn_cma_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_patch_cnn_cma_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_patch_cnn_cma_vit.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_dual_patch_cnn_cma_vit.add_argument("--act", type=str, default='selu', help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--depth_block4", type=int, default=4, help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--gamma_cma", type=float, default=0.5, help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--division_lr", type=int, default=0, help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--classifier", type=str, default='mlp', help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--in_vit_channels", type=int, default=64, help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--init_type", type=str, default='xavier_uniform', help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--gamma_crossattn_patchtrans", type=float, default=-1, help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--patch_crossattn_resolution", type=str, default='1-2', help="in_size=8, in_channels=112")
    parser_dual_patch_cnn_cma_vit.add_argument("--gamma_self_patchtrans", type=float, default=-1, help="")
    parser_dual_patch_cnn_cma_vit.add_argument("--patch_self_resolution", type=str, default='1-2', help="in_size=16, in_channels=80")
    parser_dual_patch_cnn_cma_vit.add_argument("--rm_ff", type=int, default=1, help="")

    parser_pairwise_dual_patch_cnn_cma_vit = sub_parser.add_parser('pairwise_dual_patch_cnn_cma_vit', help='My model')
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--weight_importance", type=float, default=2.0)
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--margin", type=float, default=2.0)
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--flatten_type",type=str, default="channel", required=False, help="Type of backbone")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--patch_size",type=int, default=2, required=False, help="Type of backbone")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--act", type=str, default='relu', help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--depth_block4", type=int, default=2, help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--gamma_cma", type=float, default=-1, help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--division_lr", type=int, default=0, help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--classifier", type=str, default='mlp', help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--in_vit_channels", type=int, default=64, help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--init_type", type=str, default='xavier_uniform', help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--embedding_return", type=str, default='mlp_hidden')
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--gamma_crossattn_patchtrans", type=float, default=-1, help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--patch_crossattn_resolution", type=str, default='1-2', help="in_size=8, in_channels=112")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--gamma_self_patchtrans", type=float, default=-1, help="")
    parser_pairwise_dual_patch_cnn_cma_vit.add_argument("--patch_self_resolution", type=str, default='1-2', help="in_size=16, in_channels=80")

    parser_pairwise_dual_cnn_vit = sub_parser.add_parser('pairwise_dual_cnn_vit', help='My model')
    parser_pairwise_dual_cnn_vit.add_argument("--weight_importance", type=float, default=2.0)
    parser_pairwise_dual_cnn_vit.add_argument("--margin", type=float, default=2.0)
    parser_pairwise_dual_cnn_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_pairwise_dual_cnn_vit.add_argument("--version",type=str, default="ca-fadd-0.8", required=False, help="Some changes in model")
    parser_pairwise_dual_cnn_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_pairwise_dual_cnn_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_pairwise_dual_cnn_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_pairwise_dual_cnn_vit.add_argument("--normalize_ifft", type=str, default='batchnorm', help="Normalize after ifft")
    parser_pairwise_dual_cnn_vit.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_pairwise_dual_cnn_vit.add_argument("--conv_attn", type=int, default=0, help="")   
    parser_pairwise_dual_cnn_vit.add_argument("--ratio", type=int, default=1, help="")   
    parser_pairwise_dual_cnn_vit.add_argument("--qkv_embed", type=int, default=1, help="")   
    parser_pairwise_dual_cnn_vit.add_argument("--inner_ca_dim", type=int, default=0, help="") 
    parser_pairwise_dual_cnn_vit.add_argument("--init_ca_weight", type=int, default=1, help="") 
    parser_pairwise_dual_cnn_vit.add_argument("--prj_out", type=int, default=0, help="")
    parser_pairwise_dual_cnn_vit.add_argument("--act", type=str, default='relu', help="")
    parser_pairwise_dual_cnn_vit.add_argument("--position_embed", type=int, default=1, help="")
    parser_pairwise_dual_cnn_vit.add_argument("--init_weight", type=int, default=0, help="")
    parser_pairwise_dual_cnn_vit.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_pairwise_dual_cnn_vit.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_pairwise_dual_cnn_vit.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_pairwise_dual_cnn_vit.add_argument("--division_lr", type=int, default=0, help="")
    parser_pairwise_dual_cnn_vit.add_argument("--classifier", type=str, default='vit_aggregate_-1', help="")
    parser_pairwise_dual_cnn_vit.add_argument("--embedding_return", type=str, default='mlp_hidden', help="")

    parser_ori_dual_eff_vit = sub_parser.add_parser('origin_dual_efficient_vit', help='My model')
    parser_ori_dual_eff_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_ori_dual_eff_vit.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_ori_dual_eff_vit.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_ori_dual_eff_vit.add_argument("--freeze", type=int, default=0,  help="Weight for frequency vectors")
    parser_ori_dual_eff_vit.add_argument("--pretrained", type=int, default=0, required=True, help="Weight for frequency vectors")
    parser_ori_dual_eff_vit.add_argument("--division_lr", type=int, default=0, help="")

    parser_ori_dual_eff_vit_rm_ifft = sub_parser.add_parser('origin_dual_efficient_vit_remove_ifft', help='My model')
    parser_ori_dual_eff_vit_rm_ifft.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_ori_dual_eff_vit_rm_ifft.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_ori_dual_eff_vit_rm_ifft.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_ori_dual_eff_vit_rm_ifft.add_argument("--freeze", type=int, default=0,  help="Weight for frequency vectors")
    parser_ori_dual_eff_vit_rm_ifft.add_argument("--pretrained", type=int, default=0, required=True, help="Weight for frequency vectors")
    parser_ori_dual_eff_vit_rm_ifft.add_argument("--division_lr", type=int, default=0, help="")

    parser_dual_cnn_feedforward_vit = sub_parser.add_parser('dual_cnn_feedforward_vit', help='My model')
    parser_dual_cnn_feedforward_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_cnn_feedforward_vit.add_argument("--aggregation",type=str, default="add-0.8", required=False, help="Some changes in model")
    parser_dual_cnn_feedforward_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_cnn_feedforward_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_cnn_feedforward_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_cnn_feedforward_vit.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_cnn_feedforward_vit.add_argument("--conv_reduction_channels", type=int, default=0, help="")   
    parser_dual_cnn_feedforward_vit.add_argument("--ratio_reduction", type=int, default=1, help="")   
    parser_dual_cnn_feedforward_vit.add_argument("--input_freq_dim", type=int, default=88, help="")   
    parser_dual_cnn_feedforward_vit.add_argument("--hidden_freq_dim", type=int, default=256, help="")   
    parser_dual_cnn_feedforward_vit.add_argument("--position_embed", type=int, default=1, help="")
    parser_dual_cnn_feedforward_vit.add_argument("--init_weight", type=int, default=0, help="")
    parser_dual_cnn_feedforward_vit.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_dual_cnn_feedforward_vit.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_dual_cnn_feedforward_vit.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_dual_cnn_feedforward_vit.add_argument("--division_lr", type=int, default=0, help="")

    parser_dual_pairwise_cnn_feedforward_vit = sub_parser.add_parser('pairwise_dual_cnn_feedforward_vit', help='My model')
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--weight_importance", type=float, default=2.0)
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--margin", type=float, default=2.0)
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--aggregation",type=str, default="add-0.8", required=False, help="Some changes in model")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--conv_reduction_channels", type=int, default=0, help="")   
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--ratio_reduction", type=int, default=1, help="")   
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--input_freq_dim", type=int, default=88, help="")   
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--hidden_freq_dim", type=int, default=256, help="")   
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--position_embed", type=int, default=1, help="")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--init_weight", type=int, default=0, help="")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--init_linear", type=str, default="xavier", help="")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--init_layernorm", type=str, default="normal", help="")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--init_conv", type=str, default="kaiming", help="")
    parser_dual_pairwise_cnn_feedforward_vit.add_argument("--division_lr", type=int, default=0, help="")
    
    ############# adjust image
    parser.add_argument('--adj_brightness',type=float, default = 1, help='adj_brightness')
    parser.add_argument('--adj_contrast',type=float, default = 1, help='adj_contrast')

    return parser.parse_args()

import torch
import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 
torch.set_num_threads(2)
# try:
#     import mkl
#     mkl.set_num_threads(1)
# except:
#     pass

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    model = args.model
    # Config device
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Adjustness:
    adj_brightness = float(args.adj_brightness)
    adj_contrast = float(args.adj_contrast)
    
    # Save args to text:
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

        
    ################# TRAIN #######################
    if model == "xception":
        from module.train_torch import train_image_stream
        from model.cnn.xception_net.model import xception
        model = xception(pretrained=args.pretrained)
        args_txt = "lr{}_batch{}_es{}_loss{}_pre{}_seed{}".format(args.lr, args.batch_size, args.es_metric, args.loss, args.pretrained, args.seed)
        args_txt += "_drmlp{}_aug{}".format(0.0, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="xception", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "xception_rfm":
        from module.train_rfm import train_rfm
        from model.cnn.xception_net.model_two_out import xception
        model = xception(pretrained=args.pretrained)
        args_txt = "lr{}_batch{}_es{}_lossce_pre{}_seed{}".format(args.lr, args.batch_size, args.es_metric,args.pretrained, args.seed)
        args_txt += "_drmlp{}_aug{}".format(0.0, args.augmentation)
            
        train_rfm(model, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="xception", args_txt=args_txt, augmentation=args.augmentation)
    
    elif model == 'capsule':
        from module.train_two_outclass import train_capsulenet
        args_txt = "lr{}_batch{}_es{}_beta{}_dropout{}_seed{}_drmlp0.0_aug{}".format(args.lr, args.batch_size, args.es_metric, args.beta, args.dropout, args.seed, args.augmentation)
        train_capsulenet(train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, gpu_id=args.gpu_id, beta1=args.beta, dropout=args.dropout, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="capsulenet", args_txt=args_txt, augmentation=args.augmentation)
           
    elif model == 'srm_two_stream':
        from module.train_torch import train_image_stream
        from model.cnn.srm_two_stream.twostream import Two_Stream_Net
        
        model = Two_Stream_Net()
        args_txt = "lr{}_batch{}_es{}_loss{}_seed{}".format(args.lr, args.batch_size, args.es_metric, args.loss, args.seed)
        args_txt += "_drmlp{}_aug{}".format(0.0, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="srm_2_stream", args_txt=args_txt, augmentation=args.augmentation)
        
    elif model == "meso4":
        from model.cnn.mesonet4.model import mesonet
        from module.train_torch import train_image_stream
        model = mesonet(image_size=args.image_size)
        args_txt = "lr{}_batch{}_es{}_loss{}_seed{}".format(args.lr, args.batch_size, args.es_metric, args.loss, args.seed)
        args_txt += "_drmlp{}_aug{}".format(0.0, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="meso4", args_txt=args_txt, augmentation=args.augmentation)
        
    elif model == "dual_efficient":
        from module.train_torch import train_dual_stream
        from model.cnn.dual_cnn.dual_efficient import DualEfficient
        
        model = DualEfficient(pretrained=args.pretrained)
        args_txt = "lr{}_batch{}_es{}_loss{}_pre{}_seed{}".format(args.lr, args.batch_size, args.es_metric,args.loss, args.pretrained, args.seed)
        args_txt += "_drmlp{}_aug{}".format(0.0, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr, division_lr=False, use_pretrained=False,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_efficient", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "dual_attn_cnn":
        from module.train_torch import train_dual_stream
        from model.cnn.dual_attn_cnn.model import DualCrossAttnCNN
    
        model = DualCrossAttnCNN(image_size=args.image_size, num_classes=1, dim=args.dim, mlp_dim=args.mlp_dim,\
                                backbone=args.backbone, pretrained=bool(args.pretrained), unfreeze_blocks=args.unfreeze_blocks,\
                                normalize_ifft=args.normalize_ifft,\
                                flatten_type=args.flatten_type, patch_size=args.patch_size,\
                                conv_attn=bool(args.conv_attn), ratio=args.ratio, qkv_embed=bool(args.qkv_embed), inner_ca_dim=args.inner_ca_dim, init_ca_weight=bool(args.init_ca_weight), prj_out=bool(args.prj_out), act=args.act,\
                                version=args.version, 
                                init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                dropout_in_mlp=args.dropout_in_mlp)
        
        args_txt = "lr{}-{}_batch{}_es{}_loss{}_v{}_pool{}_bb{}_pre{}_unf{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.version, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "norm{}_".format(args.normalize_ifft)
        args_txt += "flat{}_patch{}_".format(args.flatten_type, args.patch_size)
        args_txt += "convattn{}_r{}_qkvemb{}_incadim{}_prj{}_act{}_".format(args.conv_attn, args.ratio, args.qkv_embed, args.inner_ca_dim, args.prj_out, args.act)
        args_txt += "dim{}_mlpdim{}_".format(args.dim, args.mlp_dim)
        if args.init_weight == 1:
            args_txt += "init_{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=False,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_attn_cnn", args_txt=args_txt, augmentation=args.augmentation)
        
    elif model == "efficient_vit":
        from module.train_torch import train_image_stream
        from model.vision_transformer.cnn_vit.efficient_vit import EfficientViT

        dropout = 0.15
        emb_dropout = 0.15
        model = EfficientViT(
            selected_efficient_net=0,
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            pool=args.pool, 
            dropout_in_mlp=args.dropout_in_mlp,
            pretrained=args.pretrained,
            freeze=args.freeze
        )
        args_txt = "batch{}_pool{}_lr{}-{}_patch{}_h{}_d{}_dim{}_mlpdim{}_es{}_loss{}_pre{}_freeze{}_seed{}".format(args.batch_size, args.pool, args.lr, args.division_lr, args.patch_size, args.heads, args.depth, args.dim, args.mlp_dim, args.es_metric, args.loss, args.pretrained, args.freeze, args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=False,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed, \
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="efficient_vit", args_txt=args_txt, augmentation=args.augmentation)
        
    elif model == "dual_cnn_vit":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_cnn_vit.model import DualCNNViT
        
        dropout = 0.0
        emb_dropout = 0.0
        model = DualCNNViT(image_size=args.image_size, num_classes=1, dim=args.dim,\
                                depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,\
                                dim_head=args.dim_head, dropout=0.15, \
                                backbone=args.backbone, pretrained=bool(args.pretrained),\
                                normalize_ifft=args.normalize_ifft,\
                                flatten_type=args.flatten_type,\
                                conv_attn=bool(args.conv_attn), ratio=args.ratio, qkv_embed=bool(args.qkv_embed), inner_ca_dim=args.inner_ca_dim, init_ca_weight=bool(args.init_ca_weight), prj_out=bool(args.prj_out), act=args.act,\
                                patch_size=args.patch_size, \
                                version=args.version, unfreeze_blocks=args.unfreeze_blocks, \
                                init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                dropout_in_mlp=args.dropout_in_mlp, classifier=args.classifier)
        
        args_txt = "lr{}-{}_b{}_es{}_l{}_cls{}_v_{}_d{}_md{}_h{}_d{}_p{}_bb{}_pre{}_unf{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.classifier, args.version, args.dim, args.mlp_dim, args.heads, args.depth, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "norm{}_".format(args.normalize_ifft)
        args_txt += "f{}_pat{}_".format(args.flatten_type, args.patch_size)
        args_txt += "conv{}_r{}_qkv{}_cad{}_prj{}_act{}_".format(args.conv_attn, args.ratio, args.qkv_embed, args.inner_ca_dim, args.prj_out, args.act)
        if args.init_weight == 1:
            args_txt += "init_{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_cnn_vit", args_txt=args_txt, augmentation=args.augmentation)
        
    elif model == "origin_dual_efficient_vit":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_cnn_vit.origin_dual_efficient_vit import OriginDualEfficientViT
        
        dropout = 0.15
        emb_dropout = 0.15
        model = OriginDualEfficientViT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            version=args.version,
            weight=args.weight,
            freeze=args.freeze,
            dropout_in_mlp=args.dropout_in_mlp,
            pretrained=args.pretrained,
        )
        
        args_txt = "batch{}_v{}_w{}_lr{}-{}_patch{}_mlpdim{}_dim{}_h{}_d{}_es{}_loss{}_pre{}_freeze{}_seed{}".format(args.batch_size, args.version, args.weight, args.lr, args.division_lr, args.patch_size, args.mlp_dim, args.dim, args.heads, args.depth, args.es_metric, args.loss, args.pretrained, args.freeze, args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="origin_dual_efficient_vit", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "origin_dual_efficient_vit_remove_ifft":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_cnn_vit.origin_dual_efficient_vit_without_ifft import OriginDualEfficientViTWithoutIFFT
        
        dropout = 0.15
        emb_dropout = 0.15
        model = OriginDualEfficientViTWithoutIFFT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            version=args.version,
            weight=args.weight,
            freeze=args.freeze,
            dropout_in_mlp=args.dropout_in_mlp,
            pretrained=args.pretrained
        )
        
        args_txt = "batch{}_v{}_w{}_lr{}-{}_patch{}_mlpdim{}_dim{}_h{}_d{}_es{}_loss{}_pre{}_freeze{}_seed{}".format(args.batch_size, args.version, args.weight, args.lr, args.division_lr, args.patch_size, args.mlp_dim, args.dim, args.heads, args.depth, args.es_metric, args.loss, args.pretrained, args.freeze, args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="origin_dual_efficient_vit_remove_ifft", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "pairwise_dual_cnn_vit":
        from module.train_pairwise import train_pairwise_dual_stream
        from model.vision_transformer.dual_cnn_vit.pairwise_dual_cnn_vit import PairwiseDualCNNViT

        dropout = 0.15
        emb_dropout = 0.15
        model = PairwiseDualCNNViT(image_size=args.image_size, num_classes=1, dim=args.dim,\
                                depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,\
                                dim_head=args.dim_head, dropout=0.15,\
                                backbone=args.backbone, pretrained=bool(args.pretrained),\
                                normalize_ifft=args.normalize_ifft,\
                                flatten_type=args.flatten_type,\
                                conv_attn=bool(args.conv_attn), ratio=args.ratio, qkv_embed=bool(args.qkv_embed), inner_ca_dim=args.inner_ca_dim, init_ca_weight=bool(args.init_ca_weight), prj_out=bool(args.prj_out), act=args.act,\
                                patch_size=args.patch_size, \
                                version=args.version, unfreeze_blocks=args.unfreeze_blocks, \
                                init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                dropout_in_mlp=args.dropout_in_mlp, embedding_return=args.embedding_return, classifier=args.classifier)
        
        args_txt = "lr{}-{}_b{}_es{}_l{}_cls{}_ret{}_im{}_mar{}_v{}_md{}_d{}_h{}_d{}_p{}_bb{}_pre{}_unf{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.classifier, args.embedding_return, args.weight_importance, args.margin, args.version, args.mlp_dim, args.dim, args.heads, args.depth, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "norm{}_".format(args.normalize_ifft)
        args_txt += "f{}_patch{}_".format(args.flatten_type, args.patch_size)
        args_txt += "conv{}_r{}_qkv{}_cad{}_prj{}_act{}_".format(args.conv_attn, args.ratio, args.qkv_embed, args.inner_ca_dim, args.prj_out, args.act)
        if args.init_weight == 1:
            args_txt += "init{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_pairwise_dual_stream(model, args.weight_importance, args.margin, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="pairwise_dual_cnn_vit", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "dual_cnn_feedforward_vit":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_cnnfeedforward_vit.model import DualCNNFeedForwardViT
        
        dropout = 0.15
        emb_dropout = 0.15
        model = DualCNNFeedForwardViT(image_size=args.image_size, num_classes=1, dim=args.dim,\
                                    depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,\
                                    dim_head=args.dim_head, dropout=0.15, emb_dropout=0.15,\
                                    backbone=args.backbone, pretrained=bool(args.pretrained), unfreeze_blocks=args.unfreeze_blocks,\
                                    conv_reduction_channels=args.conv_reduction_channels, ratio_reduction=args.ratio_reduction,\
                                    flatten_type=args.flatten_type, patch_size=args.patch_size,\
                                    input_freq_dim=args.input_freq_dim, hidden_freq_dim=args.hidden_freq_dim,\
                                    position_embed=bool(args.position_embed), pool=args.pool,\
                                    aggregation=args.aggregation,\
                                    init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                    dropout_in_mlp=args.dropout_in_mlp)
        
        args_txt = "lr{}-{}_batch{}_es{}_loss{}_agg{}_mlpdim{}_dim{}_h{}_d{}_pool{}_bb{}_pre{}_unf{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.aggregation, args.mlp_dim, args.dim, args.heads, args.depth, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "flat{}_patch{}_".format(args.flatten_type, args.patch_size)
        args_txt += "convredu{}_r{}_".format(args.conv_reduction_channels, args.ratio_reduction)
        args_txt += "inpffdim{}_hidffdim{}_".format(args.input_freq_dim, args.hidden_freq_dim)
        if args.init_weight == 1:
            args_txt += "init_{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed_{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_cnn_feedforward_vit", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "pairwise_dual_cnn_feedforward_vit":
        from module.train_pairwise import train_pairwise_dual_stream
        from model.vision_transformer.dual_cnnfeedforward_vit.pairwise_dual_cnn_feedfoward_vit import PairwiseDualCNNFeedForwardViT
        
        dropout = 0.15
        emb_dropout = 0.15
        model = PairwiseDualCNNFeedForwardViT(image_size=args.image_size, num_classes=1, dim=args.dim,\
                                    depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,\
                                    dim_head=args.dim_head, dropout=0.15, emb_dropout=0.15,\
                                    backbone=args.backbone, pretrained=bool(args.pretrained), unfreeze_blocks=args.unfreeze_blocks,\
                                    conv_reduction_channels=args.conv_reduction_channels, ratio_reduction=args.ratio_reduction,\
                                    flatten_type=args.flatten_type, patch_size=args.patch_size,\
                                    input_freq_dim=args.input_freq_dim, hidden_freq_dim=args.hidden_freq_dim,\
                                    position_embed=bool(args.position_embed), pool=args.pool,\
                                    aggregation=args.aggregation,\
                                    init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                    dropout_in_mlp=args.dropout_in_mlp)
        
        args_txt = "lr{}-{}_batch{}_es{}_loss{}_im{}_mar{}_agg{}_mlpdim{}_dim{}_h{}_d{}_pool{}_bb{}_pre{}_unf{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.weight_importance, args.margin, args.aggregation, args.mlp_dim, args.dim, args.heads, args.depth, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "flat{}_patch{}_".format(args.flatten_type, args.patch_size)
        args_txt += "convredu{}_r{}_".format(args.conv_reduction_channels, args.ratio_reduction)
        args_txt += "inpffdim{}_hidffdim{}_".format(args.input_freq_dim, args.hidden_freq_dim)
        if args.init_weight == 1:
            args_txt += "init_{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_pairwise_dual_stream(model, args.weight_importance, args.margin, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="pairwise_dual_cnn_feedforward_vit", args_txt=args_txt, augmentation=args.augmentation)
    
    elif model == "vit":
        from module.train_torch import train_image_stream
        from model.vision_transformer.vit.vit import ViT

        model = ViT(image_size=args.image_size, patch_size=args.patch_size, num_classes=1, dim=args.dim, depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim, pool = args.pool, channels = 3, dim_head = args.dim_head, dropout = 0., emb_dropout = 0.)
        args_txt = "batch{}_pool{}_lr{}-{}_patch{}_h{}_d{}_dim{}_mlpdim{}_es{}_loss{}_seed{}".format(args.batch_size, args.pool, args.lr, 0, args.patch_size, args.heads, args.depth, args.dim, args.mlp_dim, args.es_metric, args.loss, args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=0, use_pretrained=False,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed, \
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="vit", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "swin_vit":
        from module.train_torch import train_image_stream
        from model.vision_transformer.vit.swin_vit import *

        model = swin_t(window_size=args.window_size)
        args_txt = "batch{}_lr{}-{}_window{}_es{}_loss{}_seed{}".format(args.batch_size, args.lr, 0, args.window_size,args.es_metric, args.loss, args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=0, use_pretrained=False,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed, \
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="swin_vit", args_txt=args_txt, augmentation=args.augmentation)

    elif model == 'm2tr':
        from module.train_torch import train_image_stream
        from model.vision_transformer.m2tr.m2tr import M2TR

        model_cfg = {
            'IMG_SIZE': args.image_size,
            'BACKBONE': args.backbone,
            'TEXTURE_LAYER': args.texture_layer,
            'FEATURE_LAYER': args.feature_layer,
            'DEPTH': args.depth,
            'NUM_CLASSES': 1,
            'DROP_RATIO': args.drop_ratio,
            'HAS_DECODER': args.has_decoder
        }
        model = M2TR(model_cfg)
        args_txt = "batch{}_lr{}-{}_bb{}_texture{}_feature{}_d{}_drop{}_decoder{}_es{}_loss{}_seed{}".format(args.batch_size, args.lr, 0, args.backbone, args.texture_layer, args.feature_layer, args.depth, args.drop_ratio, args.has_decoder, args.es_metric, args.loss, args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=0, use_pretrained=False,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed, \
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="m2tr", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "dual_cnn_cma_transformer":
        from model.vision_transformer.dual_cnn_vit.modelv2 import DualCNNCMATransformer
        from module.train_torch import train_dual_stream

        model = DualCNNCMATransformer(image_size=args.image_size, num_classes=1, depth=args.depth, \
                            backbone=args.backbone, pretrained=args.pretrained, unfreeze_blocks=args.unfreeze_blocks, \
                            normalize_ifft=args.normalize_ifft,\
                            act=args.act,\
                            patch_resolution=args.patch_resolution,\
                            init_type=args.init_type,
                            mlp_dim=args.mlp_dim, dropout_in_mlp=args.dropout_in_mlp)
        
        args_txt = "lr{}-{}_batch{}_es{}_loss{}_d{}_bb{}_pre{}_unf{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.depth, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "norm{}_".format(args.normalize_ifft)
        args_txt += "act{}_".format(args.act)
        args_txt += "reso{}_".format(args.patch_resolution)
        args_txt += "init_{}_".format(args.init_type)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_cnn_cma_transformer", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "dual_cma_cnn_vit":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_cnn_vit.modelv3 import DualCMACNNViT
        
        dropout = 0.0
        emb_dropout = 0.0
        model = DualCMACNNViT(gpu_id=args.gpu_id, image_size=args.image_size, num_classes=1, dim=args.dim,\
                                depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,\
                                dim_head=args.dim_head, dropout=0.15, emb_dropout=0.15,\
                                backbone=args.backbone, pretrained=bool(args.pretrained), gamma_cma=args.gamma_cma,\
                                normalize_ifft=args.normalize_ifft,\
                                flatten_type=args.flatten_type,\
                                conv_attn=bool(args.conv_attn), ratio=args.ratio, qkv_embed=bool(args.qkv_embed), inner_ca_dim=args.inner_ca_dim, init_ca_weight=bool(args.init_ca_weight), prj_out=bool(args.prj_out), act_fusion=args.act_fusion, act_cma=args.act_cma,\
                                patch_size=args.patch_size, position_embed=bool(args.position_embed), pool=args.pool,\
                                version=args.version, unfreeze_blocks=args.unfreeze_blocks, \
                                init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                dropout_in_mlp=args.dropout_in_mlp)
        
        args_txt = "lr{}-{}_batch{}_es{}_loss{}_v{}_dim{}_mlpdim{}_h{}_d{}_pool{}_bb{}_pre{}_unf{}_gamma{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.version, args.dim, args.mlp_dim, args.heads, args.depth, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks, args.gamma_cma)
        args_txt += "norm{}_".format(args.normalize_ifft)
        args_txt += "flat{}_patch{}_".format(args.flatten_type, args.patch_size)
        args_txt += "convattn{}_r{}_qkvemb{}_incadim{}_prj{}_actf{}-c{}_".format(args.conv_attn, args.ratio, args.qkv_embed, args.inner_ca_dim, args.prj_out, args.act_fusion, args.act_cma)
        if args.init_weight == 1:
            args_txt += "init_{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}_".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_cma_cnn_vit", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "dual_cma_cnn_attn":
        from module.train_torch import train_dual_stream
        from model.cnn.dual_cma_cnn_attn.model import DualCMACNNAttn
    
        model = DualCMACNNAttn(image_size=args.image_size, num_classes=1, dim=args.dim, mlp_dim=args.mlp_dim,\
                                backbone=args.backbone, pretrained=bool(args.pretrained), unfreeze_blocks=args.unfreeze_blocks, gamma_cma=args.gamma_cma,\
                                normalize_ifft=args.normalize_ifft,\
                                flatten_type=args.flatten_type, patch_size=args.patch_size,\
                                conv_attn=bool(args.conv_attn), ratio=args.ratio, qkv_embed=bool(args.qkv_embed), inner_ca_dim=args.inner_ca_dim, init_ca_weight=bool(args.init_ca_weight), prj_out=bool(args.prj_out), act_fusion=args.act_fusion, act_cma=args.act_cma,\
                                version=args.version, 
                                init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                dropout_in_mlp=args.dropout_in_mlp)
        
        args_txt = "lr{}-{}_batch{}_es{}_loss{}_v{}_pool{}_bb{}_pre{}_unf{}_gamma{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.version, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks, args.gamma_cma)
        args_txt += "norm{}_".format(args.normalize_ifft)
        args_txt += "flat{}_patch{}_".format(args.flatten_type, args.patch_size)
        args_txt += "convattn{}_r{}_qkvemb{}_incadim{}_prj{}_actf{}-c{}_".format(args.conv_attn, args.ratio, args.qkv_embed, args.inner_ca_dim, args.prj_out, args.act_fusion, args.act_cma)
        args_txt += "dim{}_mlpdim{}_".format(args.dim, args.mlp_dim)
        if args.init_weight == 1:
            args_txt += "init{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}_".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=False,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_cma_cnn_attn", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "dual_patch_cnn_cma_vit":
        from model.vision_transformer.dual_cnn_vit.dual_patch_cnn_cma_vit import DualPatchCNNCMAViT
        from module.train_torch import train_dual_stream

        model = DualPatchCNNCMAViT(image_size=args.image_size, num_classes=1, depth_block4=args.depth_block4, \
                backbone=args.backbone, pretrained=args.pretrained, unfreeze_blocks=args.unfreeze_blocks, \
                normalize_ifft=args.normalize_ifft,\
                act=args.act,\
                init_type=args.init_type, \
                gamma_cma=args.gamma_cma, gamma_crossattn_patchtrans=args.gamma_crossattn_patchtrans, patch_crossattn_resolution=args.patch_crossattn_resolution, \
                gamma_self_patchtrans=args.gamma_self_patchtrans, patch_self_resolution=args.patch_self_resolution, flatten_type='patch', patch_size=2, \
                dim=args.dim, depth_vit=args.depth, heads=args.heads, dim_head=args.dim_head, dropout=0.0, emb_dropout=0.0, mlp_dim=args.mlp_dim, dropout_in_mlp=args.dropout_in_mlp, \
                classifier=args.classifier, in_vit_channels=args.in_vit_channels, rm_ff=args.rm_ff)
        
        args_txt = "lr{}-{}_b{}_es{}_l{}_bb{}_pre{}_unf{}_rmff{}_gamma{}-{}-{}_depb4{}_flat{}_pa{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.backbone, args.pretrained, args.unfreeze_blocks, args.rm_ff, args.gamma_cma,args.gamma_self_patchtrans, args.gamma_crossattn_patchtrans, args.depth_block4, args.flatten_type, args.patch_size)
        args_txt += "norm{}_selfreso{}_attnreso{}_".format(args.normalize_ifft, args.patch_self_resolution, args.patch_crossattn_resolution)
        args_txt += "act{}_".format(args.act)
        args_txt += "init{}_".format(args.init_type)
        args_txt += "seed{}_cls{}_".format(args.seed, args.classifier)
        if 'vit' in args.classifier:
            args_txt += 'd{}_md{}_d{}_h{}_'.format(args.dim, args.mlp_dim, args.depth, args.heads)
        args_txt += "drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_patch_cnn_cma_vit", args_txt=args_txt, augmentation=args.augmentation)

    elif model == "pairwise_dual_patch_cnn_cma_vit":
        from model.vision_transformer.dual_cnn_vit.pairwise_dual_patch_cnn_cma_vit import PairwiseDualPatchCNNCMAViT
        from module.train_pairwise import train_pairwise_dual_stream

        model = PairwiseDualPatchCNNCMAViT(image_size=args.image_size, num_classes=1, depth_block4=args.depth_block4, \
                backbone=args.backbone, pretrained=args.pretrained, unfreeze_blocks=args.unfreeze_blocks, \
                normalize_ifft=args.normalize_ifft,\
                act=args.act,\
                init_type=args.init_type, \
                gamma_cma=args.gamma_cma, gamma_crossattn_patchtrans=args.gamma_crossattn_patchtrans, patch_crossattn_resolution=args.patch_crossattn_resolution, \
                gamma_self_patchtrans=args.gamma_self_patchtrans, patch_self_resolution=args.patch_self_resolution, flatten_type='patch', patch_size=2, \
                dim=args.dim, depth_vit=args.depth, heads=args.heads, dim_head=args.dim_head, dropout=0.0, emb_dropout=0.0, mlp_dim=args.mlp_dim, dropout_in_mlp=args.dropout_in_mlp, \
                classifier=args.classifier, in_vit_channels=args.in_vit_channels, embedding_return=args.embedding_return)
        
        args_txt = "lr{}-{}_b{}_es{}_l{}_ret{}_im{}_mar{}__bb{}_pre{}_unf{}_gamma{}-{}-{}_depb4{}_flat{}_patch{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.embedding_return, args.weight_importance, args.margin, args.backbone, args.pretrained, args.unfreeze_blocks, args.gamma_cma, args.gamma_self_patchtrans, args.gamma_crossattn_patchtrans, args.depth_block4, args.flatten_type, args.patch_size)
        args_txt += "norm{}_selfreso{}_attnreso{}_".format(args.normalize_ifft, args.patch_self_resolution, args.patch_crossattn_resolution)
        args_txt += "act{}_".format(args.act)
        args_txt += "init{}_".format(args.init_type)
        args_txt += "seed{}_cls{}_".format(args.seed, args.classifier)
        if 'vit' in args.classifier:
            args_txt += 'd{}_md{}_d{}_h{}_'.format(args.dim, args.mlp_dim, args.depth, args.heads)
        args_txt += "drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_pairwise_dual_stream(model, args.weight_importance, args.margin, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                        batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                        adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="pairwise_dual_patch_cnn_cma_vit", args_txt=args_txt, augmentation=args.augmentation)
    
    elif model == "dual_cnn_vit_test":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_cnn_vit.test import DualCNNViTTest
        
        dropout = 0.0
        emb_dropout = 0.0
        model = DualCNNViTTest(gpu_id=args.gpu_id, image_size=args.image_size, num_classes=1, dim=args.dim,\
                                depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,\
                                dim_head=args.dim_head, dropout=0.15, \
                                backbone=args.backbone, pretrained=bool(args.pretrained),\
                                normalize_ifft=args.normalize_ifft,\
                                flatten_type=args.flatten_type,\
                                conv_attn=bool(args.conv_attn), ratio=args.ratio, qkv_embed=bool(args.qkv_embed), inner_ca_dim=args.inner_ca_dim, init_ca_weight=bool(args.init_ca_weight), prj_out=bool(args.prj_out), act=args.act,\
                                patch_size=args.patch_size, position_embed=bool(args.position_embed), pool=args.pool,\
                                version=args.version, unfreeze_blocks=args.unfreeze_blocks, \
                                init_weight=args.init_weight, init_linear=args.init_linear, init_layernorm=args.init_layernorm, init_conv=args.init_conv, \
                                dropout_in_mlp=args.dropout_in_mlp, classifier=args.classifier)
        
        args_txt = "lr{}-{}_b{}_es{}_l{}_cls{}_v_{}_d{}_md{}_h{}_d{}_p{}_bb{}_pre{}_unf{}_".format(args.lr, args.division_lr, args.batch_size, args.es_metric, args.loss, args.classifier, args.version, args.dim, args.mlp_dim, args.heads, args.depth, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "norm{}_".format(args.normalize_ifft)
        args_txt += "f{}_patch{}_".format(args.flatten_type, args.patch_size)
        args_txt += "conv{}_r{}_qkv{}_cd{}_prj{}_act{}_".format(args.conv_attn, args.ratio, args.qkv_embed, args.inner_ca_dim, args.prj_out, args.act)
        if args.init_weight == 1:
            args_txt += "init{}-{}-{}_".format(args.init_linear, args.init_layernorm, args.init_conv)
        args_txt += "seed{}".format(args.seed)
        args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "_gamma{}".format(args.gamma)
            criterion.append(args.gamma)
        use_pretrained = True if args.pretrained or args.resume != '' else False
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr, division_lr=args.division_lr, use_pretrained=use_pretrained,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual_cnn_vit_test", args_txt=args_txt, augmentation=args.augmentation)
    