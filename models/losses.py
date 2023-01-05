import os
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

class CriterionLoss():
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = {}
        self.criterion["MAE"] = nn.L1Loss()
        self.criterion["MSE"] = nn.MSELoss()
        self.criterion["CE"] = nn.CrossEntropyLoss()
        self.criterion["BCE"] = nn.BCELoss()
        self.criterion["COS_SIM"] = self.cosine_similarity_loss
        self.criterion["ACC"] = self.get_ACC_cls
        self.criterion["R2"] = self.get_R2_reg
        self.reset_metric()
        self.set_key2loss()

        self.draw_log = False
        if "DRAW_LOG" in cfg and cfg["DRAW_LOG"]:
            self.draw_log = True
            self.draw_log_dir = cfg["DRAW_LOG_DIR"]
            self.reset_draw_log()

    def save_log_graph(self, datas, save_name):
        xs = np.arange(1, len(datas)+1)
        plt.plot(xs, datas, "-o")
        plt.savefig(save_name, dpi=500)
        plt.clf()

    def save_draw_log(self):
        for mode in self.log_dict:
            for key in self.log_dict[mode]:
                # save LOSS
                for func in self.log_dict[mode][key]["loss"]:
                    # save all
                    save_name = os.path.join(self.draw_log_dir, "{}_{}_LOSS_{}_all.png".format(mode, key, func))
                    self.save_log_graph(self.log_dict[mode][key]["loss"][func]["all"], save_name)
                    # save epoch
                    save_name = os.path.join(self.draw_log_dir, "{}_{}_LOSS_{}_epoch.png".format(mode, key, func))
                    self.save_log_graph(self.log_dict[mode][key]["loss"][func]["epoch"], save_name)
                # save METRIC
                for func in self.log_dict[mode][key]["metric"]:
                    # save all
                    save_name = os.path.join(self.draw_log_dir, "{}_{}_METRIC_{}_all.png".format(mode, key, func))
                    self.save_log_graph(self.log_dict[mode][key]["metric"][func]["all"], save_name)
                    # save epoch
                    save_name = os.path.join(self.draw_log_dir, "{}_{}_METRIC_{}_epoch.png".format(mode, key, func))
                    self.save_log_graph(self.log_dict[mode][key]["metric"][func]["epoch"], save_name)

    def update_draw_log_epoch(self):
        for mode in self.log_dict:
            for key in self.log_dict[mode]:
                # update LOSS
                for func in self.log_dict[mode][key]["loss"]:
                    self.log_dict[mode][key]["loss"][func]["all"] += self.log_dict[mode][key]["loss"][func]["tmp"]
                    avg_val = sum(self.log_dict[mode][key]["loss"][func]["tmp"]) / len(self.log_dict[mode][key]["loss"][func]["tmp"])
                    self.log_dict[mode][key]["loss"][func]["epoch"].append(avg_val)
                    self.log_dict[mode][key]["loss"][func]["tmp"] = []
                # update METRIC
                for func in self.log_dict[mode][key]["metric"]:
                    self.log_dict[mode][key]["metric"][func]["all"] += self.log_dict[mode][key]["metric"][func]["tmp"]
                    avg_val = sum(self.log_dict[mode][key]["metric"][func]["tmp"]) / len(self.log_dict[mode][key]["metric"][func]["tmp"])
                    self.log_dict[mode][key]["metric"][func]["epoch"].append(avg_val)
                    self.log_dict[mode][key]["metric"][func]["tmp"] = []

    def reset_draw_log(self):
        self.log_dict = {}
        for mode in ["train", "test"]:
            log_dict = {}
            for key in self.metrics:
                log_dict[key] = {}
                log_dict[key]["loss"] = {}
                for func in self.metrics[key]["loss"]:
                    log_dict[key]["loss"][func] = {"all": [], "epoch": [], "tmp": []}
                log_dict[key]["loss"]["SUM"] = {"all": [], "epoch": [], "tmp": []}
                log_dict[key]["metric"] = {}
                for func in self.metrics[key]["metric"]:
                    log_dict[key]["metric"][func] = {"all": [], "epoch": [], "tmp": []}
                log_dict[key]["metric"]["SUM"] = {"all": [], "epoch": [], "tmp": []}
            self.log_dict[mode] = log_dict
    
    def cosine_similarity_loss(self, output, gt):
        return torch.mean(1.0 - F.cosine_similarity(output, gt))

    def set_key2loss(self):
        self.key2loss = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
            if datas["DATA"] == "META":
                key = "{}_{}".format(datas["DATA"], data_idx)
            elif datas["DATA"] == "TIME":
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            elif datas["DATA"] == "CYCLE":
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            losses = datas["LOSS"]
            if isinstance(losses, str): losses = [losses]
            self.key2loss[key] = losses

    def reset_metric(self):
        self.metrics = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
            if datas["DATA"] == "META":
                key = "{}_{}".format(datas["DATA"], data_idx)
            elif datas["DATA"] == "TIME":
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            elif datas["DATA"] == "CYCLE":
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            self.add_metric_with_key(key, datas["LOSS"], datas["METRIC"])
            
    def add_metric_with_key(self, key, losses, metrics):
        metric_dict = {}
        metric_dict["num_data"] = 0
        metric_dict["loss_all"] = 0
        metric_dict["loss_avg"] = 0
        metric_dict["loss_last"] = 0
        metric_dict["loss"] = {}
        if isinstance(losses, str): losses = [losses]
        for loss_func in losses:
            metric_dict["loss"][loss_func] = {"sum":0, "avg":0}
        metric_dict["metric_all"] = 0
        metric_dict["metric_avg"] = 0
        metric_dict["metric_last"] = 0
        metric_dict["metric"] = {}
        if isinstance(metrics, str): metrics = [metrics]
        for metric in metrics:
            metric_dict["metric"][metric] = {"sum":0, "avg":0}
        self.metrics[key] = metric_dict
    
    def get_R2_reg(self, output, gt, multioutput="uniform_average"):
        if isinstance(output, torch.Tensor): output = output.cpu().numpy()
        if isinstance(gt, torch.Tensor): gt = gt.cpu().numpy()
        if len(output.shape) > 1 and output.shape[0] == 1: output = output[0]
        if len(gt.shape) > 1 and gt.shape[0] == 1: gt = gt[0]
        R2 = r2_score(gt, output, multioutput=multioutput)
        return R2

    def get_ACC_cls(self, output, gt):
        _, pred = torch.max(output, 1)
        num_correct = torch.sum(pred == gt.data)
        acc = num_correct / float(len(gt))
        return acc

    def get_losses(self, outputs, gts):
        loss_sum = None
        for key, output in outputs.items():
            loss_funcs = self.key2loss[key]
            for loss_func in loss_funcs:
                criterion = self.criterion[loss_func]
                if loss_func == "CE": gts[key] = gts[key].long()
                loss = criterion(output, gts[key])
                loss_sum = loss if loss_sum is None else loss+loss_sum
                # print(key, loss_func, loss.item())
        return loss_sum

    def update_metric(self, outputs, gts, mode="train"):
        # print(self.metrics)
        for key, output in outputs.items():
            gt = gts[key]
            self.metrics[key]["num_data"] += len(gt)
            # update LOSS
            loss_sum = None
            loss_funcs = self.key2loss[key]
            with torch.no_grad():
                for loss_func in loss_funcs:
                    if loss_func == "CE": gt = gt.long()
                    loss = self.criterion[loss_func](output, gt).cpu().numpy().item()
                    self.metrics[key]["loss"][loss_func]["sum"] += loss
                    self.metrics[key]["loss"][loss_func]["avg"] = self.metrics[key]["loss"][loss_func]["sum"] \
                                                                  / self.metrics[key]["num_data"]
                    loss_sum = loss if loss_sum is None else loss+loss_sum
                    # logging for draw
                    if self.draw_log:
                        self.log_dict[mode][key]["loss"][loss_func]["tmp"].append(loss)

            self.metrics[key]["loss_all"] += loss_sum
            self.metrics[key]["loss_avg"] = self.metrics[key]["loss_all"] \
                                            / self.metrics[key]["num_data"]
            self.metrics[key]["loss_last"] = loss_sum
            # logging for draw
            if self.draw_log:
                self.log_dict[mode][key]["loss"]["SUM"]["tmp"].append(loss)

            # update METIRCS
            metric_sum = None
            metric_funcs = list(self.metrics[key]["metric"].keys())
            with torch.no_grad():
                for metric_func in metric_funcs:
                    if metric_func == "CE": gt = gt.long()
                    metric = self.criterion[metric_func](output, gt)
                    if isinstance(metric, torch.Tensor): metric = metric.cpu().numpy().item()
    
                    self.metrics[key]["metric"][metric_func]["sum"] += metric
                    self.metrics[key]["metric"][metric_func]["avg"] = self.metrics[key]["metric"][metric_func]["sum"] \
                                                                      / self.metrics[key]["num_data"]
                    # logging for draw
                    if self.draw_log:
                        self.log_dict[mode][key]["metric"][metric_func]["tmp"].append(metric)
                    if metric_func not in ["ACC", "R2"]: metric = 1 - metric
                    metric_sum = metric if metric_sum is None else metric+metric_sum
            self.metrics[key]["metric_all"] += metric_sum
            self.metrics[key]["metric_avg"] = self.metrics[key]["metric_all"] \
                                              / self.metrics[key]["num_data"]
            self.metrics[key]["metric_last"] = metric_sum
            # logging for draw
            if self.draw_log:
                self.log_dict[mode][key]["metric"]["SUM"]["tmp"].append(metric_sum)



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

