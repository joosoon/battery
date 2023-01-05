import argparse
import os
from torch.nn.modules import module
from tqdm import tqdm
import numpy as np
import yaml

from models import load_model, CriterionLoss
from models import CosineAnnealingWarmUpRestarts
from loaders import load_dataset

import torch
import torch.nn as nn
import torchvision.transforms as transforms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battery Performace')
    parser.add_argument('--cfg', type=str, default="tmp_211009")
    # param for dataset 
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--k_fold', type=int, default=None, 
                        help="9-fold cross-validation (0~8) or None if split evenly")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    # param for model
    parser.add_argument('--ckp_file', type=str, default=None, help="path/to/trained/weight")
    parser.add_argument('--resume', action="store_true")
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--device', default='cuda', choices=["cuda", "cpu"])
    # param for backward
    parser.add_argument('--loss', type=str, default="MSE", choices=["BCE", "MSE"])
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    # logging
    parser.add_argument('--draw_log', action='store_true')
    parser.add_argument('--save_root', type=str, default=None,
                        help="PATH/TO/SAVE/CKPS, from CFG if None")
    args = parser.parse_args()

    # set device (GPU and CPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else: device = torch.device("cpu")

    # device = torch.device("cpu")

    # load config file
    if args.cfg[-5:] != '.yaml': args.cfg = args.cfg + '.yaml'
    if not os.path.isfile(args.cfg): args.cfg = './cfgs/' + args.cfg
    with open(args.cfg, encoding='UTF8') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        args.cfg = args.cfg[:-5]

    # set seed for reproduciblity
    seed = cfg["DATASET"].get("SEED", args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # save directory
    save_dir = args.save_root if args.save_root else cfg["SAVE_ROOT"]
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, "{}_kfold_{}_best.pth".format(os.path.basename(args.cfg), args.k_fold))

    # load dataset
    train_dataset, coll_fn = load_dataset(cfg, args.k_fold, "train")
    test_dataset, _ = load_dataset(cfg, args.k_fold, "test")
    train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset, batch_size=args.batch_size, 
                        num_workers=args.num_workers, shuffle=True, 
                        collate_fn=coll_fn)
    test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=1, 
                        num_workers=args.num_workers, shuffle=False, 
                        collate_fn=coll_fn)

    # load model
    model = load_model(cfg, device)
    if args.ckp_file:
        # model.load_state_dict(torch.load(args.ckp_file)['net'])
        ckp = torch.load(args.ckp_file, map_location=torch.device('cpu'))
        model.load_ckp(ckp['net'])

    # load criterion
    if args.draw_log:
        cfg["DRAW_LOG"] = True
        cfg["DRAW_LOG_DIR"] = os.path.join(save_dir, "log", "kfold_{}".format(args.k_fold))
        os.makedirs(cfg["DRAW_LOG_DIR"], exist_ok=True)

    criterion = CriterionLoss(cfg)

    # get params
    params = model.get_params() 
    # set optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.001, T_up=10, gamma=0.5)

    # set training epoch
    START_EPOCH = 0
    if args.resume and args.ckp_file:
        START_EPOCH = torch.load(args.ckp_file)['epoch']
        # set learning rate with scheduler
        for i in range(START_EPOCH):
            scheduler.step()    
            lr_curr = optimizer.param_groups[0]['lr']

    best_loss, best_metric = None, None
    for epoch in tqdm(range(START_EPOCH, args.max_epoch)):
        print("---" * 20)
        print("[EPOCH ({}/{})]".format(epoch, args.max_epoch))
        
        # train 
        model.set_mode("train")
        criterion.reset_metric()
        for idx, (data_input, data_gt) in enumerate(train_loader):
            # forward
            data_input = {k:v.to(device) for k, v in data_input.items()}
            data_gt = {k:v.to(device) for k, v in data_gt.items()}
            outputs = model(data_input)
            # compute loss
            loss_sum = criterion.get_losses(outputs, data_gt)
            # backward
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            # logging
            criterion.update_metric(outputs, data_gt, mode="train")
            print_str = ""
            for k, v in criterion.metrics.items():
                print_str += "[{}] ".format(k)
                print_str += "loss: {:.4f} ({:.4f}) | ".format(v["loss_avg"], v["loss_last"])
                print_str += "metric: {:.4f} ({}/{})| ".format(v["metric_avg"], str(v["metric_all"])[:4], v["num_data"])
            # print("[Epoch {}] TRAINING".format(epoch), print_str, end="\r")
            # print("+++ TRAIN|", criterion.metrics)
        # print()
        print("... TRAINING |", print_str)
        # print("---" * 20)

        # eval
        model.set_mode("test")
        criterion.reset_metric()
        for idx, (data_input, data_gt) in enumerate(test_loader):
            # forward
            data_input = {k:v.to(device) for k, v in data_input.items()}
            data_gt = {k:v.to(device) for k, v in data_gt.items()}
            with torch.no_grad():
                outputs = model(data_input)
            # logging
            criterion.update_metric(outputs, data_gt, mode="test")
            print_str = ""
            for k, v in criterion.metrics.items():
                print_str += "[{}] ".format(k)
                print_str += "loss: {:.4f} ({:.4f}) | ".format(v["loss_avg"], v["loss_last"])
                print_str += "metric: {:.4f} ({}/{})| ".format(v["metric_avg"], str(v["metric_all"])[:4], v["num_data"])
            # print("[Epoch {}] EVALUATE".format(epoch), print_str, end="\r")
            # print("+++ EVAL |", criterion.metrics)
        # print()
        print("... EVALUATE |", print_str)

        # drawing log
        if criterion.draw_log:
            criterion.update_draw_log_epoch()

        # save model with saving criterion (lower LOSS or higher METRIC)
        sum_loss = sum([v["loss_avg"] for k, v in criterion.metrics.items()])
        sum_metric = sum([v["metric_avg"] for k, v in criterion.metrics.items()])
        if not best_loss or sum_loss < best_loss:
            best_loss = sum_loss
            save_name = os.path.join(save_dir, "{}_kfold_{}_best_loss.pth".format(os.path.basename(args.cfg), args.k_fold))
            state = {'net': model.get_ckp(), 'epoch': epoch, 
                     'loss': sum_loss, 'metric': sum_metric}
            torch.save(state, save_name)
        if not best_metric or sum_metric > best_metric:
            best_metric = sum_metric
            save_name = os.path.join(save_dir, "{}_kfold_{}_best_metric.pth".format(os.path.basename(args.cfg), args.k_fold))
            state = {'net': model.get_ckp(), 'epoch': epoch, 
                     'loss': sum_loss, 'metric': sum_metric}
            torch.save(state, save_name)
        # save latest model
        save_name = os.path.join(save_dir, "{}_kfold_{}_latest.pth".format(os.path.basename(args.cfg), args.k_fold))
        state = {'net': model.get_ckp(), 'epoch': epoch, 
                 'loss': sum_loss, 'metric': sum_metric}
        torch.save(state, save_name)

        # step learning rate scheduler
        scheduler.step()


    # drawing log
    if criterion.draw_log:
        criterion.save_draw_log()
        