import os, sys

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, tunning_metric="val_loss"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.cooldown = patience
        self.best_score = sys.float_info.max if 'loss' in tunning_metric else 0.0
        self.early_stop = False
        self.tunning_metric = tunning_metric

    def better(self, cur_score):
        if "loss" in self.tunning_metric:
            return bool(cur_score < self.best_score)
        else:
            return bool(cur_score > self.best_score)
        
    def __call__(self, cur_score):
        if self.tunning_metric == 'none':
            return
        
        if not self.better(cur_score):
            self.cooldown -= 1
            if self.cooldown == 0:
                self.early_stop = True
        else:
            if self.verbose:
                print("### BEST ###: {} {}: ({:.6f} --> {:.6f})".format(self.tunning_metric, 'decreased' if 'loss' in self.tunning_metric else 'increased',\
                                                                        self.best_score, cur_score))
            self.best_score = cur_score
            self.counter = self.patience

    def save_checkpoint(self, val_loss, model, args):
        """Saves model when validation loss decreases or accuracy/f1 increases."""
        if self.verbose:
            if args.tuning_metric == "loss":
                print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {args.model_dir}")
            else:
                print(
                    f"{args.tuning_metric} increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {args.model_dir}"
                )
        model.save_pretrained(args.model_dir)
        torch.save(args, os.path.join(args.model_dir, "training_args.bin"))
        self.val_loss_min = val_loss