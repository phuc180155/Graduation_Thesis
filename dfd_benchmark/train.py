import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import torch.nn as nn
import argparse
import json
# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_set', default="data/train/", help='path to train data ')
    parser.add_argument('--val_set', default="data/test/", help='path to test data ')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None,required=True, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=str, default = '', help='Resume from checkpoint ')
    parser.add_argument('--print_every',type=int, default = 5000, help='Print evaluate info every step train')
    parser.add_argument('--loss',type=str, default = "bce", help='Loss function use')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,resnext50, resnext ,gan,meso,xception')

    ## torch
    parser_capsule = subparsers.add_parser('capsule', help='Capsule')
    parser_capsule.add_argument("--seed",type=int,required=False,default=0,help="Manual seed")
    parser_capsule.add_argument("--beta1",type=int,required=False,default=0.9,help="Manual seed")
    parser_drn = subparsers.add_parser('drn', help='DRN  ')
    parser_local_nn = subparsers.add_parser('local_nn', help='Local NN ')
    parser_self_attention = subparsers.add_parser('self_attention', help='Self Attention ')

    parser_resnext50 = subparsers.add_parser('resnext50', help='Resnext50 ')
    parser_resnext101 = subparsers.add_parser('resnext101', help='Resnext101 ')
    parser_myresnext = subparsers.add_parser('myresnext', help='My Resnext ')
    parser_mnasnet = subparsers.add_parser('mnasnet', help='mnasnet pytorch ')
    parser_xception_torch = subparsers.add_parser('xception_torch', help='Xception pytorch ')
    parser_xception2_torch = subparsers.add_parser('xception2_torch', help='Xception2 pytorch ')
    parser_dsp_fwa = subparsers.add_parser('dsp_fwa', help='DSP_SWA pytorch ')
    parser_siamese_torch = subparsers.add_parser('siamese_torch', help='Siamese pytorch ')
    parser_siamese_torch.add_argument("--length_embed",type=int,required=False,default=1024,help="Length of embed vector")
    parser_meso = subparsers.add_parser('meso4_torch', help='Mesonet4')


    parser_pairwise = subparsers.add_parser('pairwise', help='Pairwises pytorch ')
    parser_pairwise.add_argument("--mode",type=int,required=True,default=0,help="0: train siamese net, 1: train classify net ")
    parser_pairwise.add_argument("--pair_path",type=str,required=False,default="pairwise_0.pt",help="Path to pairwise network ")

    parser_pairwise_efficient = subparsers.add_parser('pairwise_efficient', help='Pairwises Efficient pytorch ')
    parser_pairwise_efficient.add_argument("--mode",type=int,required=True,default=0,help="0: train siamese net, 1: train classify net ")
    parser_pairwise_efficient.add_argument("--pair_path",type=str,required=False,default="pairwise_0.pt",help="Path to pairwise network ")


    parser_gan = subparsers.add_parser('gan', help='GAN fingerprint')
    parser_gan.add_argument("--total_train_img",type=float,required=False,default=10000,help="Total image in training set")
    parser_gan.add_argument("--total_val_img",type=int,required=False,default=2000,help="Total image in testing set")

    # parser_afd.add_argument('--depth',type=int,default=10, help='AFD depth linit')
    # parser_afd.add_argument('--min',type=float,default=0.1, help='minimum_support')
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')
    parser_wavelet = subparsers.add_parser('wavelet', help='Wavelet Net')
    parser_wavelet = subparsers.add_parser('waveletnoatt', help='WaveletNoAtt Net')
    parser_wavelet = subparsers.add_parser('wavelet_res', help='WaveletRes Net')
    parser_normal = subparsers.add_parser('normal', help='Normal Wavelet Net')


    #################################################################
    ################ VIT
    #################################################################
    parser.add_argument('--dim',type=int, default = 1024, help='dim of embeding')
    parser.add_argument('--depth',type=int, default = 6, help='Number of attention layer in transformer module')
    parser.add_argument('--heads',type=int, default = 8, help='number of head in attention layer')
    parser.add_argument('--mlp_dim',type=int, default = 2048, help='dim of hidden layer in transformer layer')
    parser.add_argument('--dim_head',type=int, default = 64, help='in transformer layer ')

    parser_vit = subparsers.add_parser('vit', help='ViT transformer Net')
    parser_crossvit = subparsers.add_parser('crossvit', help='CrossViT transformer Net')
    parser_efficientvit = subparsers.add_parser('efficientvit', help='CrossViT transformer Net')
    parser_efficientvit.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_cross_efficientvit = subparsers.add_parser('crossefficientvit', help='CrossViT transformer Net')
    parser_cross_efficientvit.add_argument("--sm_patch_size",type=int,default=7,help="patch_size in cross vit")
    parser_cross_efficientvit.add_argument("--lg_patch_size",type=int,default=56,help="patch_size in cross vit")
    parser_waddvit = subparsers.add_parser('waddvit', help='CrossViT transformer and WADD')
    parser_waddvit.add_argument("--selected_block",type=int,default=5,help="patch_size in cross vit")
    parser_waddvit.add_argument("--patch_size",type=int,default=4,help="patch_size in cross vit")
    parser_cross_waddvit = subparsers.add_parser('crosswaddvit', help='CrossViT transformer and WADD')
    parser_cross_waddvit.add_argument("--selected_sm_block",type=int,default=5,help="patch_size in cross vit")
    parser_cross_waddvit.add_argument("--selected_lg_block",type=int,default=1,help="patch_size in cross vit")
    parser_cross_waddvit.add_argument("--sm_patch_size",type=int,default=8,help="patch_size in cross vit")
    parser_cross_waddvit.add_argument("--lg_patch_size",type=int,default=16,help="patch_size in cross vit")
    parser_swim_vit = subparsers.add_parser('swimvit', help='Swim transformer ')

    parser_multires_waddvit = subparsers.add_parser('multireswaddvit', help=' Multires  ViT transformer and WADD')

    parser_efficient = subparsers.add_parser('efficient', help='Efficient Net')
    parser_efficient.add_argument("--type",type=str,required=False,default="0",help="Type efficient net 0-8")
    parser_efficientdual = subparsers.add_parser('efficientdual', help='Efficient Net')
    parser_efft = subparsers.add_parser('efft', help='Efficient Net fft')
    parser_efft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    parser_e4dfft = subparsers.add_parser('e4dfft', help='Efficient Net 4d fft')
    parser_e4dfft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")
    ## tf
    parser_meso = subparsers.add_parser('meso4', help='Mesonet4')
    parser_xception_tf = subparsers.add_parser('xception_tf', help='Xceptionnet tensorflow')
    parser_siamese_tf = subparsers.add_parser('siamese_tf', help='siamese tensorflow')

    parser_srm_twostream = subparsers.add_parser('srm', help='SRM')

    ##############  gc
    parser_spectrum = subparsers.add_parser('spectrum', help='siamese tensorflow')
    parser_headpose = subparsers.add_parser('heapose', help='siamese tensorflow')
    parser_visual = subparsers.add_parser('visual', help='siamese tensorflow')

    parser_dual_eff_vit_v4 = subparsers.add_parser('dual_efficient_vit_v4', help='My model')
    parser_dual_eff_vit_v4.add_argument("--patch_size",type=int,default=7,help="patch_size in vit")
    parser_dual_eff_vit_v4.add_argument("--version",type=str, default="cross_attention-freq-add", required=True, help="Some changes in model")
    parser_dual_eff_vit_v4.add_argument("--weight", type=float, default=1, help="Weight for frequency vectors")
    parser_dual_eff_vit_v4.add_argument("--pretrained", type=int, default=0, help="")
    parser_dual_eff_vit_v4.add_argument("--architecture", type=str, default='xception_net', help="Weight for frequency vectors")

    ## adjust image
    parser.add_argument('--adj_brightness',type=float, default = 1, help='adj_brightness')
    parser.add_argument('--adj_contrast',type=float, default = 1, help='adj_contrast')

    return parser.parse_args()

def get_criterion_torch(arg_loss):
    criterion = None
    if arg_loss == "bce":
        criterion = nn.BCELoss()
    elif arg_loss == "focal":
        from pytorch_model.focal_loss import FocalLoss
        criterion = FocalLoss(gamma=2)
    return criterion

def get_loss_tf(arg_loss):
    loss = 'binary_crossentropy'
    if arg_loss == "bce":
        loss = 'binary_crossentropy'
    elif arg_loss == "focal":
        from tf_model.focal_loss import BinaryFocalLoss
        loss = BinaryFocalLoss(gamma=2)
    return loss
if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    adj_brightness = float(args.adj_brightness)
    adj_contrast = float(args.adj_contrast)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    with open(os.path.join(args.checkpoint, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if model== "capsule":
        from pytorch_model.train_torch import train_capsule
        train_capsule(train_set = args.train_set,val_set = args.val_set,gpu_id=gpu_id,manualSeed=args.seed,resume=args.resume,beta1=args.beta1, \
                      dropout=0.05,image_size=args.image_size,batch_size=args.batch_size,lr=args.lr, \
                      num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter,\
                      adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "drn":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.drn.drn_seg import DRNSub
        model = DRNSub(1)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "local_nn":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.local_nn import local_nn
        model = local_nn()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "self_attention":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.self_attention import self_attention
        model = self_attention()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "resnext50":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import resnext50
        model = resnext50()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "resnext101":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import resnext101
        model = resnext101()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "myresnext":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import MyResNetX
        model = MyResNetX()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "mnasnet":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import mnasnet
        model = mnasnet()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "xception_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.xception import xception
        model = xception(pretrained=True)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "xception2_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.xception import xception2
        model = xception2(pretrained=True)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "meso4_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import mesonet
        model = mesonet(image_size=args.image_size)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "dsp_fwa":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.DSP_FWA.models.classifier import SPPNet
        model = SPPNet(backbone=50, num_class=1)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "siamese_torch":
        from pytorch_model.train_torch import train_siamese
        from pytorch_model.siamese import SiameseNetworkResnet
        model = SiameseNetworkResnet(length_embed = args.length_embed,pretrained=True)
        train_siamese(model,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,length_embed = args.length_embed,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "pairwise":
        from pytorch_model.pairwise.train_pairwise import train_pairwise
        from pytorch_model.pairwise.model import Pairwise,ClassifyFull
        if args.mode == 0:
            model = Pairwise(args.image_size)
            train_pairwise(model,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                      batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                      epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        else:
            from pytorch_model.train_torch import train_cnn
            import torch
            model = ClassifyFull(args.image_size)
            model.cffn.load_state_dict(torch.load(os.path.join(args.checkpoint, args.pair_path)))
            criterion = get_criterion_torch(args.loss)
            train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                      image_size=args.image_size, resume=args.resume, \
                      batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                      epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "pairwise_efficient":
        from pytorch_model.efficientnet.train_pairwise import train_pairwise
        from pytorch_model.efficientnet.model_pairwise import EfficientPairwise,EfficientFull
        if args.mode == 0:
            model = EfficientPairwise()
            train_pairwise(model,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                      batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                      epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        else:
            from pytorch_model.train_torch import train_cnn
            import torch
            model = EfficientFull()
            model.efficient.load_state_dict(torch.load(os.path.join(args.checkpoint, args.pair_path)))
            criterion = get_criterion_torch(args.loss)
            train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                      image_size=args.image_size, resume=args.resume, \
                      batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                      epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "efficient":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b'+args.type,num_classes=1)
        model = nn.Sequential(model,nn.Sigmoid())
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "efficientdual":
        from pytorch_model.train_torch import train_dualcnn
        from pytorch_model.efficientnet import EfficientDual

        model = EfficientDual()
        criterion = get_criterion_torch(args.loss)
        train_dualcnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "dual_efficient_vit_v4":
        from pytorch_model.train_torch import train_dualcnn
        from pytorch_model.dual_efficient_vit_v4 import DualEfficientViTv4

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
            pool=args.pool,
            architecture=args.architecture,
            pretrained=args.pretrained
        )
        criterion = get_criterion_torch(args.loss)
        train_dualcnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "efft":
        from pytorch_model.train_torch import train_fftcnn
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=1)
        model = nn.Sequential(model, nn.Sigmoid())
        criterion = get_criterion_torch(args.loss)
        train_fftcnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "e4dfft":
        from pytorch_model.train_torch import train_4dfftcnn
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=4)
        model = nn.Sequential(model, nn.Sigmoid())
        criterion = get_criterion_torch(args.loss)
        train_4dfftcnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "wavelet":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.wavelet_model.model_wavelet import WaveletModel

        model = WaveletModel(in_channel=3)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "waveletnoatt":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.wavelet_model.model_wavelet_noatt import WaveletModelNoAtt

        model = WaveletModelNoAtt(in_channel=3)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "wavelet_res":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.wavelet_model.model_wavelet_res import WaveletResModel

        model = WaveletResModel(in_channel=3)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "normal":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.wavelet_model.model_normal import NormalModel

        model = NormalModel(in_channel=3)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "vit":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.transformer.model_vit import ViT

        model = ViT(
            image_size=args.image_size,
            patch_size=64,
            num_classes=1,
            dim=512,
            depth=6,
            heads=16,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )
        model = nn.Sequential(model, nn.Sigmoid())
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "crossvit":
        from pytorch_model.transformer.crossvit import CrossViT
        from pytorch_model.train_torch import train_cnn

        model = CrossViT(image_size = args.image_size, channels=3, num_classes=1,patch_size_small=14, patch_size_large=16)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
    elif model == "efficientvit":
        from pytorch_model.transformer.efficient_vit import EfficientViT
        from pytorch_model.train_torch import train_cnn

        model = EfficientViT(image_size = args.image_size,patch_size=int(args.patch_size))
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
    elif model == "crossefficientvit":
        from pytorch_model.transformer.cross_efficient_net import CrossEfficientViT
        from pytorch_model.train_torch import train_cnn

        model = CrossEfficientViT(image_size = args.image_size,sm_patch_size=args.sm_patch_size,lg_patch_size=args.lg_patch_size)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
    elif model == "waddvit":
        from pytorch_model.transformer.wadd_vit import WADDViT
        from pytorch_model.train_torch import train_cnn

        model = WADDViT(image_size = args.image_size,selected_block = args.selected_block,patch_size= args.patch_size)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
    elif model == "crosswaddvit":
        from pytorch_model.transformer.cross_wadd_net import CrossWADDViT
        from pytorch_model.train_torch import train_cnn

        model = CrossWADDViT(image_size = args.image_size,selected_sm_block = args.selected_sm_block,selected_lg_block=args.selected_lg_block,sm_patch_size= args.sm_patch_size,lg_patch_size=args.lg_patch_size)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
    elif model == "multireswaddvit":
        from pytorch_model.transformer.multires_wadd_vit import MultiresWADDViT
        from pytorch_model.train_torch import train_cnn

        model = MultiresWADDViT(image_size = args.image_size,
                                dim=args.dim,heads=args.heads,depth=args.depth,mlp_dim=args.mlp_dim,
                                dim_head=args.dim_head)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
    elif model == "swimvit":
        from pytorch_model.transformer.swim_transformer import swin_t
        from pytorch_model.train_torch import train_cnn

        model = swin_t()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
    elif model == "srm":
        from pytorch_model.srm_twostream.twostream import Two_Stream_Net
        from pytorch_model.train_torch import train_cnn

        model = Two_Stream_Net()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
        pass
# ---------------------------------------------------------------------------------------------
    elif model == "gan":
        from tf_model.train_tf import train_gan
        train_gan(train_set = args.train_set,val_set = args.val_set,training_seed=0,\
                  image_size=args.image_size,batch_size=args.batch_size,num_workers=args.workers, \
                  epochs=args.niter,checkpoint=args.checkpoint,total_train_img = args.total_train_img,total_val_img = args.total_val_img, \
                adj_brightness = adj_brightness, adj_contrast = adj_contrast)
        # train_gan()
        pass
    elif model == "meso4":
        from tf_model.mesonet.model import Meso4
        from tf_model.train_tf import train_cnn
        model = Meso4(image_size=args.image_size).model
        loss = get_loss_tf(args.loss)
        train_cnn(model,loss=loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter, \
                  adj_brightness=adj_brightness, adj_contrast=adj_contrast)
        pass
    elif model == "xception_tf":
        from tf_model.train_tf import train_cnn
        from tf_model.model_cnn_keras import xception
        model = xception(image_size=args.image_size)
        loss = get_loss_tf(args.loss)
        train_cnn(model,loss=loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batchSize,num_workers=1,checkpoint=args.checkpoint,epochs=args.niter, \
                  adj_brightness=adj_brightness, adj_contrast=adj_contrast)
        pass
    elif model == "siamese_tf":
        from tf_model.siamese import get_siamese_model
        from tf_model.train_tf import train_siamese
        model = get_siamese_model((args.image_size, args.image_size, 3))
        loss = 'binary_crossentropy'
        train_siamese(model,loss = loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter, \
                      adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    ###############
    elif model == "spectrum":
        from feature_model.spectrum.train_spectrum import train_spectrum

        train_spectrum(args.train_set,model_file=args.checkpoint + args.resume)
        pass
    elif model == "headpose":
        from feature_model.headpose_forensic.train_headpose import train_headpose
        train_headpose(args.train_set,model_file=args.checkpoint + args.resume)
        pass
    elif model == "visual":
        from feature_model.visual_artifact.train_visual import train_visual
        train_visual(args.train_set,model_file=args.checkpoint + args.resume)
        pass

