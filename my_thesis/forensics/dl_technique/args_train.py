import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import torch
import torch.nn as nn
import argparse
import json
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_dir', type=str, default="", help="path to train data")
    parser.add_argument('--val_dir', type=str, default="", help="path to validation data")
    parser.add_argument('--test_dir', type=str, default="", help="path to test data")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=0, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None,required=True, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=str, default = '', help='Resume from checkpoint ')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loss',type=str, default = "bce", help='Loss function use')
    parser.add_argument('--gamma',type=float, default=0.0, help="gamma hyperparameter for focal loss")
    parser.add_argument('--eval_per_iters',type=int, default=-1, help='Evaluate per some iterations')
    parser.add_argument('--es_metric',type=str, default='val_loss', help='Criterion for early stopping')
    parser.add_argument('--es_patience',type=int, default=5, help='Early stopping epoch')
    
    sub_parser = parser.add_subparsers(dest="model", help="Choose model of the available ones:  xception, efficient_dual, ViT, CrossViT, efficient ViT, dual efficient vit...")
    
    ######################## CNN architecture:
    parser_capsule = sub_parser.add_parser('capsule', help='CapsuleNet')
    parser_capsule.add_argument("--beta",type=int,required=False,default=0.9,help="Beta for optimizer Adam")
    parser_capsule.add_argument("--dropout", type=float, required=False, default=0.05)
    
    parser_xception = sub_parser.add_parser('xception', help='XceptionNet')
    parser_xception.add_argument('--pretrained', type=int, default=0)
    parser_meso4 = sub_parser.add_parser('meso4', help='MesoNet')
    parser_dual_eff = sub_parser.add_parser('dual_efficient', help="Efficient-Frequency Net")
    parser_srm_2_stream = sub_parser.add_parser('srm_two_stream', help="SRM 2 stream net from \"Generalizing Face Forgery Detection with High-frequency Features (CVPR 2021).\"")
    # Ablation study
    parser_dual_attn_eff = sub_parser.add_parser('dual_attn_efficient', help="Ablation Study")
    parser_dual_attn_eff.add_argument("--patch_size",type=int,default=7,help="patch_size")
    parser_dual_attn_eff.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_dual_attn_eff.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_dual_attn_eff.add_argument("--freeze", type=int, default=0, help="Freeze backbone")
    
    ######################## Vision transformer architecture:
    parser.add_argument('--dim',type=int, default = 1024, help='dim of embeding')
    parser.add_argument('--depth',type=int, default = 6, help='Number of attention layer in transformer module')
    parser.add_argument('--heads',type=int, default = 8, help='number of head in attention layer')
    parser.add_argument('--mlp_dim',type=int, default = 2048, help='dim of hidden layer in transformer layer')
    parser.add_argument('--dim_head',type=int, default = 64, help='in transformer layer ')
    parser.add_argument('--pool',type=str, default = "cls", help='in transformer layer ')
    
    # ViT
    parser_vit = sub_parser.add_parser('vit', help='ViT transformer Net')
    # Efficient ViT (CViT)
    parser_efficientvit = sub_parser.add_parser('efficient_vit', help='CrossViT transformer Net')
    parser_efficientvit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_efficientvit.add_argument("--freeze", type=int, default=0, help="Freeze backbone")
    # SwinViT
    parser_swim_vit = sub_parser.add_parser('swin_vit', help='Swim transformer')
    
    # My refined model:
    parser_dual_eff_vit = sub_parser.add_parser('dual_efficient_vit', help='My model')
    parser_dual_eff_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_eff_vit.add_argument("--version",type=str, default="ca-fadd-0.8", required=False, help="Some changes in model")
    parser_dual_eff_vit.add_argument("--backbone",type=str, default="efficient_net", required=False, help="Type of backbone")
    parser_dual_eff_vit.add_argument("--pretrained",type=int, default=1, required=False, help="Load pretrained backbone")
    parser_dual_eff_vit.add_argument("--unfreeze_blocks", type=int, default=-1, help="Unfreeze blocks in backbone")
    parser_dual_eff_vit.add_argument("--normalize_ifft", type=int, default=1, help="Normalize after ifft")
    parser_dual_eff_vit.add_argument("--flatten_type", type=str, default='patch', help="in ['patch', 'channel']")
    parser_dual_eff_vit.add_argument("--conv_attn", type=int, default=0, help="")   
    parser_dual_eff_vit.add_argument("--ratio", type=int, default=1, help="")   
    parser_dual_eff_vit.add_argument("--qkv_embed", type=int, default=1, help="")   
    parser_dual_eff_vit.add_argument("--inner_ca_dim", type=int, default=0, help="") 
    parser_dual_eff_vit.add_argument("--init_ca_weight", type=int, default=1, help="") 
    parser_dual_eff_vit.add_argument("--prj_out", type=int, default=0, help="")
    parser_dual_eff_vit.add_argument("--act", type=str, default='relu', help="")
    parser_dual_eff_vit.add_argument("--position_embed", type=int, default=1, help="")
 
    parser_dual_eff_vit_v2 = sub_parser.add_parser('dual_efficient_vit_v2', help='My model')
    parser_dual_eff_vit_v2.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_eff_vit_v2.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_dual_eff_vit_v2.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_dual_eff_vit_v2.add_argument("--freeze", type=int, default=0, help="Weight for frequency vectors")
    parser_dual_eff_vit_v2.add_argument("--architecture", type=str, default='xception_net', help="Weight for frequency vectors")
    
    ############# adjust image
    parser.add_argument('--adj_brightness',type=float, default = 1, help='adj_brightness')
    parser.add_argument('--adj_contrast',type=float, default = 1, help='adj_contrast')

    return parser.parse_args()

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
        from model.cnn.xception import xception
        model = xception(pretrained=args.pretrained)
        args_txt = "lr_{}_batch_{}_es_{}_loss_{}_pre_{}_seed_{}".format(args.lr, args.batch_size, args.es_metric, args.loss, args.pretrained, args.seed)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="exception", args_txt=args_txt)
    
    elif model == 'capsule':
        from module.train_two_outclass import train_capsulenet
        args_txt = "lr_{}_batch_{}_es_{}_beta_{}_dropout_{}_seed_{}".format(args.lr, args.batch_size, args.es_metric, args.beta, args.dropout, args.seed)
        train_capsulenet(train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, gpu_id=args.gpu_id, beta1=args.beta, dropout=args.dropout, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="capsulenet", args_txt=args_txt)
           
    elif model == 'srm_two_stream':
        from module.train_torch import train_image_stream
        from model.cnn.srm_two_stream.twostream import Two_Stream_Net
        
        model = Two_Stream_Net()
        args_txt = "lr_{}_batch_{}_es_{}_loss_{}_seed_{}".format(args.lr, args.batch_size, args.es_metric, args.loss, args.seed)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="srm_2_stream", args_txt=args_txt)
        
    elif model == "meso4":
        from model.cnn.mesonet import mesonet
        from module.train_torch import train_image_stream
        model = mesonet(image_size=args.image_size)
        args_txt = "lr_{}_batch_{}_es_{}_loss_{}_seed_{}".format(args.lr, args.batch_size, args.es_metric, args.loss, args.seed)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="meso4", args_txt=args_txt)
        
    elif model == "dual_efficient":
        from module.train_torch import train_dual_stream
        from model.cnn.dual_efficient import DualEfficient
        
        model = DualEfficient()
        args_txt = "lr_{}_batch_{}_es_{}_loss_{}_seed_{}".format(args.lr, args.batch_size, args.es_metric,args.loss,args.seed)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
            
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-efficient", args_txt=args_txt)

    elif model == "dual_attn_efficient":
        from module.train_torch import train_dual_stream
        from model.cnn.dual_crossattn_efficient import DualCrossAttnEfficient
        
        dropout = 0.15
        model = DualCrossAttnEfficient(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=1,
            dim=args.dim,
            mlp_dim=args.mlp_dim,
            dropout=dropout,
            version=args.version,
            weight=args.weight,
            freeze=args.freeze
        )
        
        args_txt = "batch_{}_v_{}_w_{}_lr_{}_patch_{}_es_{}_loss_{}_freeze_{}_seed_{}".format(args.batch_size, args.version, args.weight, args.image_size, args.lr, args.patch_size, args.es_metric, args.loss, args.freeze, args.seed)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-attn-efficient", args_txt=args_txt)
        
    elif model == "efficient_vit":
        from module.train_torch import train_image_stream
        from model.vision_transformer.efficient_vit import EfficientViT

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
            pool=args.pool
        )
        args_txt = "batch_{}_pool_{}_lr_{}_patch_{}_h_{}_d_{}_es_{}_loss_{}_freeze_{}_seed_{}".format(args.batch_size, args.pool, args.lr, args.patch_size, args.heads, args.depth, args.es_metric, args.loss, args.freeze, args.seed)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_image_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="efficient-vit", args_txt=args_txt)
        
    elif model == "dual_efficient_vit":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_efficient_vit import DualEfficientViT
        
        dropout = 0.15
        emb_dropout = 0.15
        model = DualEfficientViT(image_size=args.image_size, num_classes=1, dim=args.dim,\
                                depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,\
                                dim_head=args.dim_head, dropout=0.15, emb_dropout=0.15,\
                                backbone=args.backbone, pretrained=bool(args.pretrained),\
                                normalize_ifft=args.normalize_ifft,\
                                flatten_type=args.flatten_type,\
                                conv_attn=bool(args.conv_attn), ratio=args.ratio, qkv_embed=bool(args.qkv_embed), inner_ca_dim=args.inner_ca_dim, init_ca_weight=bool(args.init_ca_weight), prj_out=bool(args.prj_out), act=args.act,\
                                patch_size=args.patch_size, position_embed=bool(args.position_embed), pool=args.pool,\
                                version=args.version, unfreeze_blocks=args.unfreeze_blocks)
        
        args_txt = "lr_{}_batch_{}_es_{}_loss_{}_v_{}_pool_{}_bb_{}_pre_{}_unf_{}_".format(args.lr, args.batch_size, args.es_metric, args.loss, args.version, args.pool, args.backbone, args.pretrained, args.unfreeze_blocks)
        args_txt += "norm_{}_".format(args.normalize_ifft)
        args_txt += "flat_{}_".format(args.flatten_type)
        args_txt += "convattn_{}_r_{}_qkvemb_{}_incadim_{}_initw_{}_prj_{}_act_{}_".format(args.conv_attn, args.ratio, args.qkv_embed, args.inner_ca_dim, args.init_ca_weight, args.prj_out, args.act)
        args_txt += "patch_{}_seed_{}".format(args.patch_size, args.seed)
        print(len(args_txt))
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-efficient-vit", args_txt=args_txt)
        
    elif model == "dual_efficient_vit_v4":
        from module.train_torch import train_dual_stream
        from model.vision_transformer.dual_efficient_vit_v4 import DualEfficientViTv4
        
        dropout = 0.15
        emb_dropout = 0.15
        model = DualEfficientViTv4(
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
            pool=args.pool,
            architecture=args.architecture,
        )
        
        args_txt = "batch_{}_v_{}_w_{}_arch_{}_pool_{}_lr_{}_patch_{}_h_{}_d_{}_es_{}_loss_{}_freeze_{}_seed_{}".format(args.batch_size, args.version, args.weight, args.architecture, args.pool, args.lr, args.patch_size, args.heads, args.depth, args.es_metric, args.loss, args.freeze, args.seed)
        criterion = [args.loss]
        if args.gamma:
            args_txt += "gamma_{}".format(args.gamma)
            criterion.append(args.gamma)
        
        train_dual_stream(model, criterion_name=criterion, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir,  image_size=args.image_size, lr=args.lr,\
                           batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume, epochs=args.n_epochs, eval_per_iters=args.eval_per_iters, seed=args.seed,\
                           adj_brightness=adj_brightness, adj_contrast=adj_contrast, es_metric=args.es_metric, es_patience=args.es_patience, model_name="dual-efficient-vit-v4", args_txt=args_txt)