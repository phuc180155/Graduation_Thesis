import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#import os, logging

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import argparse
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument("--test_dir", default="", required=True, type=str, help="path to test directory")
    parser.add_argument("--batch_size", default=16, required=True, type=int, help="batch size")
    parser.add_argument("--workers", default=4, type=int, help="number workers for dataloader")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--checkpoint", default="checkpoint", required=True, type=str)
    parser.add_argument("--resume", default="", required=True, type=str, help="Model weight in checkpoint")
    parser.add_argument("--image_size", default=256, type=int, required=True, help="input size forward to model")
    parser.add_argument('--show_time',type=bool, default = False, help='Print time ')
    ## adjust image
    parser.add_argument('--adj_brightness',type=float, default = 1, help='adj_brightness')
    parser.add_argument('--adj_contrast',type=float, default = 1, help='adj_contrast')
    
    ######## Define model:
    subparser = parser.add_subparsers(dest="model", help="Choose model from list")
    
    ##### CNN architecture:
    parser_xception = subparser.add_parser('xception', help='XceptionNet')
    parser_dual_eff = subparser.add_parser('efficient_dual', help="Efficient-Frequency Net")
    
    ##### Vision transformer architecture:
    parser.add_argument('--dim',type=int, default = 1024, help='dim of embeding')
    parser.add_argument('--depth',type=int, default = 6, help='Number of attention layer in transformer module')
    parser.add_argument('--heads',type=int, default = 8, help='number of head in attention layer')
    parser.add_argument('--mlp_dim',type=int, default = 2048, help='dim of hidden layer in transformer layer')
    parser.add_argument('--dim_head',type=int, default = 64, help='in transformer layer ')
    
    # ViT
    parser_vit = subparser.add_parser('vit', help='ViT transformer Net')
    # Efficient ViT (CViT)
    parser_efficientvit = subparser.add_parser('efficient_vit', help='CrossViT transformer Net')
    parser_efficientvit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    # SwinViT
    parser_swim_vit = subparser.add_parser('swin_vit', help='Swim transformer')
    
    # My refined model:
    parser_dual_eff_vit = subparser.add_parser('dual_efficient_vit', help='My model')
    parser_dual_eff_vit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_eff_vit.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_dual_eff_vit.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    
    return parser.parse_args()

def get_criterion_torch(arg_loss):
    criterion = None
    if arg_loss == "bce":
        criterion = nn.BCELoss()
    elif arg_loss == "focal":
        from loss import FocalLoss
        criterion = FocalLoss(gamma=2)
    return criterion

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    ################# EVALUATION #######################
    if model == "xception":
        from module.eval_torch import eval_image_stream
        from model.cnn.xception import xception
        model = xception(pretrained=False)
        eval_image_stream(model, criterion=get_criterion_torch("bce"), test_dir=args.test_dir, image_size=args.image_size,\
                          batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume,\
                          adj_brightness=float(args.adj_brightness), adj_contrast=float(args.adj_contrast), show_time=args.show_time, model_name="exception", args_txt="")
    
    elif model == "efficient_dual":
        from module.eval_torch import eval_dual_stream
        from model.cnn.efficient_dual import EfficientDual
        
        model = EfficientDual()
        eval_dual_stream(model, criterion=get_criterion_torch("bce"), test_dir=args.test_dir, image_size=args.image_size,\
                          batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume,\
                          adj_brightness=float(args.adj_brightness), adj_contrast=float(args.adj_contrast), show_time=args.show_time, model_name="efficient_dual", args_txt="")
        
    elif model == "efficient_vit":
        from module.eval_torch import eval_image_stream
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
        )
        eval_image_stream(model, criterion=get_criterion_torch("bce"), test_dir=args.test_dir, image_size=args.image_size,\
                        batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume,\
                        adj_brightness=float(args.adj_brightness), adj_contrast=float(args.adj_contrast), show_time=args.show_time, model_name="efficient_vit", args_txt="")
        
    elif model == "dual_efficient_vit":
        from module.eval_torch import eval_dual_stream
        from model.vision_transformer.dual_efficient_vit import DualEfficientViT
        
        dropout = 0.15
        emb_dropout = 0.15
        model = DualEfficientViT(
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
            weight=args.weight
        )
        eval_dual_stream(model, criterion=get_criterion_torch("bce"), test_dir=args.test_dir, image_size=args.image_size,\
                          batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, resume=args.resume,\
                          adj_brightness=float(args.adj_brightness), adj_contrast=float(args.adj_contrast), show_time=args.show_time, model_name="efficient_dual_vit", args_txt="")
