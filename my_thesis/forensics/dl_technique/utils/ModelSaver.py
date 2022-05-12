import os, sys
from os.path import join

import numpy as np
import torch
from typing import List

class ModelSaver:
    """A model saver for mutiples metric"""

    def __init__(self, save_metrics: List[str]):
        """
        Args:
            save_metrics (List[str]): multiple metric for save model
                                    Example: ["val_loss", "val_acc", "test_loss", "test_acc", "test_real_f1", "test_fake_f1", "test_avg_f1"]

        """
        self.save_metrics = save_metrics
        self.best_scores = [sys.float_info.max if 'loss' in metric else 0.0 for metric in save_metrics]
        
    def better(self, cur_score: float, best_score: float, metric: str):
        if "loss" in metric:
            return bool(cur_score < best_score)
        else:
            return bool(cur_score > best_score)
        
        
    def __call__(self, iter: int, cur_scores: List[float], checkpoint_dir: str, model: torch.nn.Module):
        """
        Args:
            cur_scores (List[float]): must be respective to save_metrics.
                Example: ["val_loss", "val_acc", "test_loss", 'test_acc', "test_realf1", "test_fakef1", "test_avgf1"]
        """
        assert len(cur_scores) == len(self.save_metrics), "Number of scores must be equal with number of save metrics!!!"
        for idx in range(len(cur_scores)):
            cur_score = cur_scores[idx]
            best_score = self.best_scores[idx]
            metric = self.save_metrics[idx]
            if self.better(cur_score, best_score, metric):
                # print("Better...")
                self.save_checkpoint(iter, checkpoint_dir, model, metric, cur_score)
                self.best_scores[idx] = cur_score
            
    def save_checkpoint(self, iter: int, checkpoint_dir: str, model: torch.nn.Module, metric: str, score: float):
        cnt = 0
        for ckcpoint in os.listdir(checkpoint_dir):
            if metric in ckcpoint:
                cnt += 1
                os.remove(join(checkpoint_dir, ckcpoint))
                torch.save(model.state_dict(), join(checkpoint_dir, "best_{}_{}_{:.6f}.pt".format(metric, iter, score)))
        if cnt == 0:
            torch.save(model.state_dict(), join(checkpoint_dir, "best_{}_{}_{:.6f}.pt".format(metric, iter, score)))            
        if cnt == 2:
            print("Seem to be wrong. 2 checkpoint in one folder.")
            
    def save_last_model(self, checkpoint_dir: str, model: torch.nn.Module, iteration: int):
        cnt = 0
        for ckcpoint in os.listdir(checkpoint_dir):
            if "model_last" in ckcpoint:
                cnt += 1
                os.remove(join(checkpoint_dir, ckcpoint))
                torch.save(model.state_dict(), join(checkpoint_dir, "_model_last_{}_.pt".format(iteration)))
        if cnt == 0:
            torch.save(model.state_dict(), join(checkpoint_dir, "_model_last_{}_.pt".format(iteration)))            
        if cnt == 2:
            print("Seem to be wrong. 2 checkpoint in one folder.")
