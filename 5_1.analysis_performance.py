
import argparse
import os
from tqdm import tqdm
import numpy as np
import yaml
import matplotlib.pyplot as plt

import torch
from loader.batterycell import collate_batterycell, BatteryCellData
from model.custom import CUSTOM_FUSION
from model.loss import CriterionLoss
class ParserCRATE():
    def __init__(self, cfg):
        self.crates = ['0.1C','0.2C','0.5C','1C','2C','3C','5C']
        self.cfg = cfg
        self.output_idx2crate = {}
        for datas in self.cfg["DATASET"]["OUTPUT"]:
            name  = datas['NAME']
            if name not in self.output_idx2crate:
                self.output_idx2crate[name] = {}
            items = datas['ITEM']
            for idx, item in enumerate(items):
                self.output_idx2crate[name][idx] = item

    def parsing(self, data_dict):
        parse_dict = {}
        for k, v in data_dict.items():
            for data_idx in range(v.shape[1]):
                data_crate = self.output_idx2crate[k][data_idx]
                data_type = "_".join(data_crate.split("_")[:-1])
                crate = data_crate.split("_")[-1]
                if data_type not in parse_dict:
                    parse_dict[data_type] = {}
                parse_dict[data_type][crate] = v[:, data_idx]
        return parse_dict
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battery Performace')
    parser.add_argument('--cfg', type=str, default="config/base.yaml")
    # param for dataset 
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dataset_mode', type=str, default='test')
    # param for model
    parser.add_argument('--ckp_mode', type=str, default='latest', help="name/of/trained/weight")
    parser.add_argument('--ckp_file', type=str, default=None, help="path/to/trained/weight")
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cuda', choices=["cuda", "cpu"])
    # parma for visualization
    parser.add_argument('--vis_r2', action='store_true')
    parser.add_argument('--save_np', action='store_true')
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
    if not args.cfg.endswith('.yaml'): args.cfg += '.yaml'
    assert os.path.isfile(args.cfg), "no config of {}".format(args.cfg)
    with open(args.cfg, encoding='UTF8') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        args.cfg = args.cfg[:-5]

    # set seed for reproduciblity
    seed = cfg["DATASET"].get("SEED", args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # load dataset
    coll_fn = collate_batterycell
    test_dataset = BatteryCellData(cfg, args.dataset_mode, seed)
    test_loader  = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=1, 
                        num_workers=0, shuffle=False, 
                        collate_fn=coll_fn)

    # load model
    model = CUSTOM_FUSION(cfg, device)
    if args.ckp_file is None:
        args.ckp_file = os.path.join(cfg["SAVE_ROOT"], "{}.pth".format(args.ckp_mode))
    ckp = torch.load(args.ckp_file, map_location=torch.device('cpu'))
    model.load_ckp(ckp['net'])
    
    # load criterion
    criterion = CriterionLoss(cfg)

    prs_all = {}
    gts_all = {}
    for idx, (data_input, data_gt) in enumerate(test_loader):
        # forward
        data_input = {k:v.to(device) for k, v in data_input.items()}
        data_gt = {k:v.to(device) for k, v in data_gt.items()}
        with torch.no_grad():
            outputs = model(data_input)
        # keep output
        for k in outputs.keys():
            output = outputs[k].clone().detach().cpu()
            if k not in prs_all:
                prs_all[k] = output
            else:
                prs_all[k] = torch.cat((prs_all[k], output))
        # keep gt
        for k in data_gt.keys():
            gt = data_gt[k].clone().detach().cpu()
            if k not in gts_all:
                gts_all[k] = gt
            else:
                gts_all[k] = torch.cat((gts_all[k], gt))

    prs_all_denorm = test_dataset.transform_output.denorm(prs_all)
    gts_all_denorm = test_dataset.transform_output.denorm(gts_all)
    
    # parsing into C-RATE
    crate_parser = ParserCRATE(cfg)
    prs_parsing = crate_parser.parsing(prs_all_denorm)
    gts_parsing = crate_parser.parsing(gts_all_denorm)

    data_item = 'specific_capacity'
    
    def min_max_norm(arr):
        if isinstance(arr, np.ndarray):
            arr_norm = arr.copy()
        elif isinstance(arr, torch.Tensor):
            arr_norm = arr.clone()
        elif isinstance(arr, list):
            arr_norm = np.array(arr)
            
        max_val = arr.max()
        min_val = arr.min()
        arr_norm = (arr_norm-min_val)/(max_val-min_val)
        return arr_norm
    
    save_root = './vis'
    os.makedirs(save_root, exist_ok=True)
    
    
    ########################
    # raw capa / raw error #
    ########################
    for crate_i, crate in enumerate(crate_parser.crates):
        prs = prs_parsing[data_item][crate]
        gts = gts_parsing[data_item][crate]
        mae = np.abs(gts-prs)
        print(crate, mae.max(), mae.min(), mae.mean())
        xs = np.ones_like(gts) * (crate_i+1)
        ys = gts
        cs = np.clip(mae, a_min=0, a_max=100)
        plt.scatter(xs, ys, c=cs)
        plt.xlabel("C-Rate")
    plt.colorbar().set_label('MAE')
    plt.ylabel("Specific Capacity")
    plt.xticks(list(range(1, len(crate_parser.crates)+1)),
                labels=crate_parser.crates, fontsize=12)
    save_name = "heatmap_raw_capa_raw_error.png"
    save_file = os.path.join(save_root, save_name)
    plt.savefig(save_file, dpi=500)
    plt.clf()
    print("saved at", save_file)
    
    ###############################
    # raw capa / percentage error #
    ###############################
    for crate_i, crate in enumerate(crate_parser.crates):
        prs = prs_parsing[data_item][crate]
        gts = gts_parsing[data_item][crate]
        mae = np.abs(gts-prs)/gts*100
        print(crate, mae.max(), mae.min(), mae.mean())
        xs = np.ones_like(gts) * (crate_i+1)
        ys = gts
        cs = np.clip(mae, a_min=0, a_max=100)
        plt.scatter(xs, ys, c=cs)
        plt.xlabel("C-Rate")
    plt.colorbar().set_label('percentage error', rotation=270)
    plt.ylabel("Specific Capacity")
    plt.xticks(list(range(1, len(crate_parser.crates)+1)),
                labels=crate_parser.crates, fontsize=12)
    save_name = "heatmap_raw_capa_perc_error.png"
    save_file = os.path.join(save_root, save_name)
    plt.savefig(save_file, dpi=500)
    plt.clf()
    print("saved at", save_file)
    
    ######################################
    # normalized capa / percentage error #
    ######################################
    # get 2d errors
    num_bin = 7
    errors = np.zeros((num_bin, len(crate_parser.crates)))
    for crate_i, crate in enumerate(crate_parser.crates):
        # get GT and predictions
        prs = prs_parsing[data_item][crate]
        gts = gts_parsing[data_item][crate]
        # get percentage error
        pce = np.abs(gts-prs)/gts*100
        # min-max normalization
        gts_norm = min_max_norm(gts)
        # split into bins
        s, e = 0, 0
        for bin_idx in range(num_bin):
            s = bin_idx * (1/num_bin)
            e = s +(1/num_bin)
            cnd_s = gts_norm >= s
            cnd_e = gts_norm < e
            cnd = np.bitwise_and(cnd_s, cnd_e)
            error = pce[cnd].mean()
            errors[num_bin-bin_idx-1, crate_i] = error
    plt.imshow(errors)
    cbar = plt.colorbar()
    cbar.set_label('percentage error', rotation=270)
    cbar.set_ticks([errors.min()+5, errors.max()-5])
    cbar.set_ticklabels(['low', 'high'])
    plt.xlabel("C-Rate")
    plt.ylabel("Specific Capacity")
    plt.xticks(list(range(len(crate_parser.crates))),
                labels=crate_parser.crates, fontsize=12)
    plt.yticks([0, 3, 6], labels=['high', 'mid', 'low'], fontsize=12)
    plt.title("Percentage Error")
    save_name = "heatmap_norm_capa_perc_error.png"
    save_file = os.path.join(save_root, save_name)
    plt.savefig(save_file, dpi=500)
    plt.clf()
    print("saved at", save_file)
    