import argparse
import os
from xml.etree.ElementTree import QName
from tqdm import tqdm
import numpy as np
import yaml
from matplotlib import pyplot as plt

from models import load_model, CriterionLoss
from loaders import load_dataset

import torch

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

    # set seed for reproduciblity
    seed = cfg["DATASET"].get("SEED", args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load dataset
    mode = "inverse_test"
    mode = "test"
    test_dataset, coll_fn = load_dataset(cfg, args.k_fold, mode)
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
    print("---" * 20)
    print("MODEL Epoch {}".format(ckp['epoch']))

    # save directory
    save_root = args.save_root
    if not save_root:
        save_root = os.path.join(
                        cfg["SAVE_ROOT"], "inverse_test",
                        os.path.basename(args.ckp_file))


    # forward and generate prediction
    cell_id2meta = {}
    cell_id2pred = {}
    for idx, (data_input, data_gt) in enumerate(test_loader):
        cell_id = test_loader.dataset.cell_ids[idx]
        # forward
        data_input = {k:v.to(device) for k, v in data_input.items()}
        with torch.no_grad():
            outputs = model(data_input)
        # keep 
        for k in data_input.keys():
            data = data_input[k].cpu().numpy()[0]
            if k not in cell_id2meta: cell_id2meta[k] = {}
            cell_id2meta[k][cell_id] = data.copy()

        # keep
        for k in outputs.keys():
            output = outputs[k].cpu().numpy()[0]
            if k not in cell_id2pred: cell_id2pred[k] = {}
            cell_id2pred[k][cell_id] = output.copy()

    # compare with desired output (=GT)
    cell_id2gt = {}
    for idx, (data_input, data_gt) in enumerate(test_loader):
        if idx >= test_dataset.num_gt: break
        cell_id = test_loader.dataset.cell_ids[idx]
        data_gt = {k:v.to(device) for k, v in data_gt.items()}
        # keep
        for k in data_gt.keys():
            data = data_gt[k].cpu().numpy()[0]
            if k not in cell_id2gt: cell_id2gt[k] = {}
            cell_id2gt[k][cell_id] = data.copy()

    print("---" * 20)
    print("... comparing PRED and GT")

    ranks = []
    for k in data_gt.keys():
        for cell_id_gt in tqdm(cell_id2gt[k].keys()):
            data_gt = cell_id2gt[k][cell_id_gt]

            xs = np.arange(1, len(data_gt)+1)
            plt.plot(xs, data_gt, "-or", label="GT")
            
            def get_meta_str(cell_id):
                if mode == "inverse_test":
                    for input_k in data_input.keys():
                        data_meta = cell_id2meta[input_k][cell_id]
                        target_ids = [v for _, v in test_dataset.target_ids.items()]
                        data_meta = list(map(str, data_meta[target_ids]))
                        data_meta_str = "|".join(data_meta)
                else: data_meta_str = ""
                return data_meta_str

            data_pred_gt = cell_id2pred[k][cell_id_gt]
            data_meta_str = get_meta_str(cell_id_gt)
            plt.plot(xs, data_pred_gt, "-ob", label="PRED({})|{}".format(cell_id_gt, data_meta_str))


            def cosine_similarity_loss(output, gt):
                import torch.nn.functional as F
                import torch
                output_t = torch.from_numpy(output)
                gt_t = torch.from_numpy(gt)
                cos_sim = F.cosine_similarity(output_t, gt_t, dim=0)
                loss = 1.0 - cos_sim
                return loss

            mae2cell_id = {}
            for cell_id_pred in cell_id2pred[k].keys():
                data_pred = cell_id2pred[k][cell_id_pred]
                mae = np.mean(np.abs(data_gt - data_pred))
                # cos = cosine_similarity_loss(data_pred, data_gt)
                # mae = mae/10 + cos
                mae2cell_id[mae] = cell_id_pred

            num_vis = 10
            rank = len(mae2cell_id)
            for tmp_i, mae in enumerate(sorted(list(mae2cell_id.keys()))[:num_vis]):
                cell_id_pred = mae2cell_id[mae]
                data_pred = cell_id2pred[k][cell_id_pred]
                data_meta_str = get_meta_str(cell_id_pred)
                plt.plot(xs, data_pred, "-o", label="{}|{}".format(cell_id_pred, data_meta_str))
                
                if cell_id_gt == cell_id_pred:
                    rank = tmp_i+1
            ranks.append(rank)

            # save graph
            save_dir = os.path.join(save_root, k, "vis")
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, "{}.png".format(cell_id_gt))
            plt.legend(loc="best")
            plt.savefig(save_name, dpi=500)
            plt.clf()
    ranks = np.array(ranks)
    print(ranks)
    k_vals = [1, 3, 5, 10]
    for k_val in k_vals:
        correct = ranks <= k_val
        num_correct = np.sum(correct)
        print("[Top {}] {:.2f}% ({}/{})".format(
              k_val, num_correct/len(ranks)*100, num_correct, len(ranks)))
