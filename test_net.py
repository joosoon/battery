import argparse
import os
from torch.nn.modules import module
from tqdm import tqdm
import numpy as np
import yaml

from models import load_model, CriterionLoss
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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    # param for model
    parser.add_argument('--ckp_file', type=str, default=None, help="path/to/trained/weight")
    parser.add_argument('--ckp_mode', type=str, default="best_metric", 
                        choices=["latest", "best_loss", "best_metric"])
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cuda', choices=["cuda", "cpu"])
    args = parser.parse_args()

    # set device (GPU and CPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else: device = torch.device("cpu")

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

    # load dataset
    test_dataset, coll_fn = load_dataset(cfg, args.k_fold, "test")
    # test_dataset, coll_fn = load_dataset(cfg, args.k_fold, "train")
    test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=1, 
                        num_workers=args.num_workers, shuffle=False, 
                        collate_fn=coll_fn)

    # load model
    if args.ckp_file is None:
        args.ckp_file = os.path.join(cfg["SAVE_ROOT"], 
                            "{}_kfold_{}_{}.pth".format(
                            os.path.basename(args.cfg), args.k_fold, args.ckp_mode))
    model = load_model(cfg, device)
    ckp = torch.load(args.ckp_file)
    model.load_ckp(ckp['net'])
    model.set_mode("test")

    # load criterion
    criterion = CriterionLoss(cfg)

    # eval
    for idx, (data_input, data_gt) in enumerate(test_loader):
        # forward
        data_input = {k:v.to(device) for k, v in data_input.items()}
        data_gt = {k:v.to(device) for k, v in data_gt.items()}
        with torch.no_grad():
            outputs = model(data_input)
        criterion.update_metric(outputs, data_gt)

    # summary
    print("---" * 20)
    print("MODEL Epoch {}".format(ckp['epoch']))
    for k, v in criterion.metrics.items():
        print("[OUTPUT]:", k)
        print("[METRIC]:", list(v["metric"].keys()))
        for metric, result in v["metric"].items():
            # print("...", metric, result["avg"]) 
            print(result["avg"], end=', ')
        print()