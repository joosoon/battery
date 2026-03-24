import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class CriterionLoss():
    def __init__(self, cfg=None):
        self.cfg = cfg
        
        self.criterion = {}
        self.criterion["SMOOTHL1"] = nn.SmoothL1Loss(beta=0.001)
        self.criterion["MAE"] = nn.L1Loss()
        self.criterion["RMSE"] = self.get_rmse
        self.criterion["MSE"] = nn.MSELoss()
        self.criterion["MAE_radian_to_degree"] = self.mae_radian_to_degree
        self.criterion["CE"] = nn.CrossEntropyLoss()
        self.criterion["BCE"] = nn.BCELoss()
        self.criterion["COS_SIM"] = self.cosine_similarity_loss
        self.criterion["R2"] = self.get_r2
        
        if self.cfg is not None:
            self.set_key2loss()
            self.set_key2metric()
        
    def set_key2loss(self):
        self.key2loss = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
            key = datas["NAME"]
            losses = datas["LOSS"]
            if isinstance(losses, str): losses = [losses]
            self.key2loss[key] = losses
            
    def set_key2metric(self):
        self.key2metric_func = {}
        self.key2metric_avg = {}
        self.buff_gts     = {}
        self.buff_outputs = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
            key = datas["NAME"]
            metrics = datas["METRIC"]
            if isinstance(metrics, str): metrics = [metrics]
            self.key2metric_func[key] = metrics
            self.key2metric_avg[key]  = {}

    def update_metrics(self, outputs, gts):
        for key, output in outputs.items():
            # keep output at buffer
            output = output.clone().detach()
            if key not in self.buff_outputs:
                self.buff_outputs[key] = output
            else:
                self.buff_outputs[key] = torch.cat((self.buff_outputs[key], output), dim=0)
            # keep gt at buffer
            gt = gts[key].clone().detach()
            if key not in self.buff_gts:
                self.buff_gts[key] = gt
            else:
                self.buff_gts[key] = torch.cat((self.buff_gts[key], gt), dim=0)

            # print(key, self.buff_outputs[key].shape, self.buff_gts[key].shape)
            for metric_func in self.key2metric_func[key]:
                criterion = self.criterion[metric_func]
                metric = criterion(self.buff_outputs[key], self.buff_gts[key])
                if metric_func != "R2": metric /= len(self.buff_outputs[key])
                if isinstance(metric, torch.Tensor): metric = metric.cpu().numpy()
                self.key2metric_avg[key][metric_func] = metric
                # print("...", metric_func, metric, self.buff_outputs[key].shape)   

    def get_r2(self, outputs, gts, multioutput="uniform_average"):
        if isinstance(outputs, torch.Tensor): outputs = outputs.cpu().numpy()
        if isinstance(gts, torch.Tensor): gts = gts.cpu().numpy()
        if len(outputs.shape) > 1: outputs = outputs.flatten()
        if len(gts.shape) > 1: gts = gts.flatten()
        R2 = r2_score(gts, outputs, multioutput=multioutput)
        return R2
    
    def get_rmse(self, outputs, gts):
        mse = self.criterion["MSE"](outputs, gts)
        return torch.sqrt(mse)
    
    def get_losses(self, outputs, gts):
        loss_sum = None
        for key, output in outputs.items():
            loss_funcs = self.key2loss[key]
            for loss_func in loss_funcs:
                criterion = self.criterion[loss_func]
                if loss_func == "CE": gts[key] = gts[key].long()
                loss = criterion(output, gts[key])
                loss_sum = loss if loss_sum is None else loss+loss_sum
        return loss_sum
                    
    def mae_radian_to_degree(self, output, gt):
        radian = self.criterion["MAE"](output, gt)
        degree = radian * 180 / math.pi
        return degree

    def cosine_similarity_loss(self, output, gt):
        return torch.mean(1.0 - F.cosine_similarity(output, gt))
    


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    