import argparse
import os
import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from models import load_model
from loaders import load_dataset

import torch
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battery Performace')
    parser.add_argument('--cfg', type=str, default="tmp_211009")
    # param for dataset 
    parser.add_argument('--dataset', type=str, default='test', choices=["train", "test"])
    parser.add_argument('--k_fold', type=int, default=None, help="9-fold cross-validation (0~8)")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    # param for model
    parser.add_argument('--ckp_file', type=str, default=None, help="path/to/trained/weight")
    parser.add_argument('--ckp_mode', type=str, default="best_metric", 
                        choices=["latest", "best_loss", "best_metric"])
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--device', default='cuda', choices=["cuda", "cpu"])
    parser.add_argument('--draw_graph', action="store_true")
    parser.add_argument('--draw_gt', action="store_true")
    parser.add_argument('--save_numpy', action="store_true")
    parser.add_argument('--save_gt', action="store_true")
    
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

    # load config file
    if args.cfg[-5:] != '.yaml': args.cfg = args.cfg + '.yaml'
    with open('./cfgs/' + args.cfg, encoding='UTF8') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        args.cfg = args.cfg[:-5]

    # load dataset
    test_dataset, coll_fn = load_dataset(cfg, args.k_fold, args.dataset)
    test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=1, 
                        num_workers=args.num_workers, shuffle=False, 
                        collate_fn=coll_fn)
                        
    # load model
    if args.ckp_file is None:
        args.ckp_file = os.path.join(cfg["SAVE_ROOT"], 
                            "{}_kfold_{}_{}.pth".format(
                            args.cfg, args.k_fold, args.ckp_mode))
    model = load_model(cfg, device)
    ckp = torch.load(args.ckp_file)
    model.load_ckp(ckp['net'])

    # save directory
    save_root = args.save_root
    if not save_root:
        # args.save_root = "./results/inference"
        # save_root = os.path.join(
        #                 args.save_root, args.cfg, 
        #                 os.path.basename(args.ckp_file), 
        #                 "kfold_{}".format(args.k_fold))
        save_root = os.path.join(
                        cfg["SAVE_ROOT"], "inference",
                        os.path.basename(args.ckp_file))

    for idx, (data_input, data_gt) in enumerate(tqdm(test_loader)):
        cell_id = test_loader.dataset.cell_ids[idx]
        # forward
        data_input = {k:v.to(device) for k, v in data_input.items()}
        with torch.no_grad():
            outputs = model(data_input)

        # save inference results as numpy
        for k in outputs.keys():
            if k == "CELLID": continue
            output = outputs[k].cpu().numpy()[0]
            gts = data_gt[k].cpu().numpy()[0]
            if args.draw_graph:
                xs = np.arange(1, len(output)+1)
                plt.plot(xs, output, "-or", label="pred")
                
                if args.draw_gt:
                    plt.plot(xs, gts, "-ob", label="GT")

                # save graph
                save_dir = os.path.join(save_root, k, "vis")
                os.makedirs(save_dir, exist_ok=True)
                save_name = os.path.join(save_dir, "{}.png".format(cell_id))
                plt.legend(loc="best")
                plt.savefig(save_name, dpi=500)
                plt.clf()

            if args.save_numpy:
                if args.save_gt:
                    output = np.stack((output, gts))
                save_dir = os.path.join(save_root, k, "numpy")
                os.makedirs(save_dir, exist_ok=True)
                save_name = os.path.join(save_dir, "{}.npy".format(cell_id))
                np.save(save_name, output)
                    
