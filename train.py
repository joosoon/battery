import argparse
import os
from tqdm import tqdm
import numpy as np
import yaml
import copy
import pickle
import random

import torch
from loader.batterycell import collate_batterycell, BatteryCellData
from model.custom import CUSTOM_FUSION, MLPOnly
from model.loss import CriterionLoss, CosineAnnealingWarmUpRestarts
from test import ParserCRATE

def get_metric(gts, prs, crate_parser):
    gts_all, prs_all = {}, {}
    for data_gt, outputs in zip(gts, prs):
        for k in data_gt.keys():
            gt = data_gt[k].clone().detach().cpu()
            gts_all[k] = gt if k not in gts_all else torch.cat((gts_all[k], gt))
            output = outputs[k].clone().detach().cpu()
            prs_all[k] = output if k not in prs_all else torch.cat((prs_all[k], output))
    prs_all_denorm = test_dataset.transform_output.denorm(prs_all)
    gts_all_denorm = test_dataset.transform_output.denorm(gts_all)
    prs_parsing = crate_parser.parsing(prs_all_denorm)
    gts_parsing = crate_parser.parsing(gts_all_denorm)
    for data_type in prs_parsing.keys():
        prs = prs_parsing[data_type]
        gts = gts_parsing[data_type]
        prs_crate_all = []
        gts_crate_all = []
        for crate in crate_parser.crates:
            if crate not in prs: continue
            prs_crate_all.append(prs[crate])
            gts_crate_all.append(gts[crate])
        prs_crate_all = torch.cat(prs_crate_all)
        gts_crate_all = torch.cat(gts_crate_all)
    last_mse = criterion.criterion["MSE"](prs_crate_all, gts_crate_all)
    last_mae = criterion.criterion["MAE"](prs_crate_all, gts_crate_all)
    last_r2  = criterion.criterion["R2" ](prs_crate_all, gts_crate_all)
    return last_mse, last_mae, last_r2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battery Performace')
    parser.add_argument('--cfg', type=str, default="config/base.yaml")
    # param for dataset 
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    # param for model
    parser.add_argument('--ckp_file', type=str, default=None, help="path/to/trained/weight")
    parser.add_argument('--resume', action="store_true")
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max_epoch', default=500, type=int)
    parser.add_argument('--device', default='cuda', choices=["cuda", "cpu"])
    # param for backward
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    # logging
    parser.add_argument('--save_root', type=str, default=None,
                        help="PATH/TO/SAVE/CKPS, from CFG if None")
    parser.add_argument('--save_interval', type=int, default=None)
    args = parser.parse_args()


    # set device (GPU and CPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else: device = torch.device("cpu")

    # load config file
    if not args.cfg.endswith('.yaml'): args.cfg += '.yaml'
    assert os.path.isfile(args.cfg)
    with open(args.cfg, encoding='UTF8') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        args.cfg = args.cfg[:-5]

    # set train parameter
    args.lr = cfg["TRINER"].get("LR", args.lr)
    args.wd = cfg["TRINER"].get("WD", args.wd)
    # args.wd = 0.1
    args.batch_size  = cfg["TRINER"].get("BATCH_SIZE", args.batch_size)
    args.num_workers = cfg["TRINER"].get("NUM_WORKER", args.num_workers)

    # set seed for reproduciblity
    seed = cfg["DATASET"].get("SEED", args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    
    # save directory
    save_dir = args.save_root if args.save_root else cfg["SAVE_ROOT"]
    os.makedirs(save_dir, exist_ok=True)

    # load dataset
    coll_fn = collate_batterycell
    train_dataset = BatteryCellData(cfg, "train", seed)
    train_loader  = torch.utils.data.DataLoader(
                        dataset=train_dataset, batch_size=args.batch_size, 
                        num_workers=args.num_workers, shuffle=True, 
                        collate_fn=coll_fn)
    
    # add validation set
    if cfg["DATASET"].get("VALSET", None) is not None:
        cfg["DATASET"]["TESTSET"] = cfg["DATASET"]["VALSET"]
        
    test_dataset = BatteryCellData(cfg, "test", seed)
    test_loader  = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=1, 
                        num_workers=args.num_workers, shuffle=False, 
                        collate_fn=coll_fn)
    # load model
    model = CUSTOM_FUSION(cfg, device)
    # model = MLPOnly(device)

    # #############################
    # # count the number of param #
    # #############################
    # pp=0
    # for p in list(model.get_params()):
    #     nn=1
    #     for s in list(p.size()):
    #         nn = nn*s
    #     pp += nn
    # print(pp)
    # exit()
    
    
    if args.ckp_file is not None:
        ckp = torch.load(args.ckp_file, map_location=torch.device('cpu'))
        model.load_ckp(ckp['net'])
        print("... load pre-trained {}".format(args.ckp_file))


    # load criterion
    criterion = CriterionLoss(cfg)

    # get params
    params = model.get_params() 
    
    # load optimizer
    optimizer = torch.optim.Adam(params, lr=args.lr/10, weight_decay=args.wd)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=args.lr, T_up=10, gamma=0.5)
    
    # ##########################
    # # draw LR with scheduler #
    # ##########################
    # import matplotlib.pyplot as plt    
    # lrs = []
    # for epoch in range(150):
    #     lr = optimizer.param_groups[0]['lr']
    #     lrs.append(lr)
    #     scheduler.step()
    # xs = list(range(len(lrs)))
    # plt.plot(xs, lrs, '-or')
    # plt.savefig("tmp_lr.png")
    # exit()
    
    log = {'mae':  {'train':[], 'test':[]},
           'mse':  {'train':[], 'test':[]},
           'r2':   {'train':[], 'test':[]}
           }
    best_mse, best_mae, best_r2 = None, None, None
    START_EPOCH = 0
    for epoch in tqdm(range(START_EPOCH, args.max_epoch)):
        print("---" * 20)
        print("Epoch {}/{}".format(epoch, args.max_epoch))
        
        # TRAINING
        gts, prs = [], []
        criterion.set_key2metric()
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
            gts.append(data_gt)
            prs.append(outputs)
            criterion.update_metrics(outputs, data_gt)
            print_str = "TRAIN - "
            for key, metrics in criterion.key2metric_avg.items():
                print_str += "[{}] ".format(key)
                for metric_func, metric_val in metrics.items():
                    print_str += "{}: {:.4f} | ".format(metric_func, metric_val)
            print_str += "{}".format(criterion.buff_outputs[key].shape)
            end_str = '\r' if idx < len(train_loader)-1 else '\n'
            print(print_str, end=end_str)

        # logging
        crate_parser = ParserCRATE(cfg)
        last_mse, last_mae, last_r2 = get_metric(gts, prs, crate_parser)
        log['mae']['train'].append(last_mse)
        log['mse']['train'].append(last_mae)
        log['r2']['train'].append(last_r2)

        # TEST
        gts, prs = [], []
        criterion.set_key2metric()
        for idx, (data_input, data_gt) in enumerate(test_loader):
            # forward
            data_input = {k:v.to(device) for k, v in data_input.items()}
            data_gt = {k:v.to(device) for k, v in data_gt.items()}
            with torch.no_grad():
                outputs = model(data_input)
            # logging
            gts.append(data_gt)
            prs.append(outputs)
            criterion.update_metrics(outputs, data_gt)
            print_str = "TEST - "
            for key, metrics in criterion.key2metric_avg.items():
                print_str += "[{}] ".format(key)
                for metric_func, metric_val in metrics.items():
                    print_str += "{}: {:.4f} | ".format(metric_func, metric_val)
            print_str += "{}".format(criterion.buff_outputs[key].shape)
            end_str = '\r' if idx < len(test_loader)-1 else '\n'
            print(print_str, end=end_str)
        print()
        
        # logging
        crate_parser = ParserCRATE(cfg)
        last_mse, last_mae, last_r2 = get_metric(gts, prs, crate_parser)
        log['mae']['test'].append(last_mse)
        log['mse']['test'].append(last_mae)
        log['r2']['test'].append(last_r2)
        
        # keep ckp and save
        if best_mse is None:
            best_mse = last_mse
        if best_mae is None:
            best_mae = last_mae
        if best_r2 is None:
            best_r2 = last_r2
            
        # keep state (ckp)
        state_last = {'net': copy.deepcopy(model.get_ckp()), 'epoch': epoch}
        if last_r2 >= best_r2:
            state_r2 = {'net': copy.deepcopy(model.get_ckp()), 'epoch': epoch}
            best_r2 = last_r2
        if last_mse <= best_mse:
            state_mse = {'net': copy.deepcopy(model.get_ckp()), 'epoch': epoch}
            best_mse = last_mse
        if last_mae <= best_mae:
            state_mae = {'net': copy.deepcopy(model.get_ckp()), 'epoch': epoch}
            best_mae = last_mae
        
        # learnig rate scheduler
        scheduler.step()
         
    # save latest model
    save_name = os.path.join(save_dir, "latest.pth")
    torch.save(state_last, save_name)
    # save best MSE
    save_name = os.path.join(save_dir, "best_mse.pth")
    torch.save(state_mse, save_name)
    # save best MAE
    save_name = os.path.join(save_dir, "best_mae.pth")
    torch.save(state_mae, save_name)
    # save best R2
    save_name = os.path.join(save_dir, "best_r2.pth")
    torch.save(state_r2, save_name)

    # save log
    save_name = os.path.join(save_dir, "log.pickle")
    with open(save_name, 'wb') as f:
        pickle.dump(log, f, pickle.HIGHEST_PROTOCOL)
    
    
    