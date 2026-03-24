
import argparse
import os
from tqdm import tqdm
import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

import torch
from loader.batterycell import collate_batterycell, BatteryCellData
from model.custom import CUSTOM_FUSION, MLPOnly
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
    parser.add_argument('--cfg_root', type=str, default="config")
    parser.add_argument('--cfg', type=str, default="base.yaml")
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
    parser.add_argument('--prt_inference', action='store_true')
    args = parser.parse_args()

    # set device (GPU and CPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else: device = torch.device("cpu")


    prs_all_fold = []
    gts_all_fold = []
    ids_all_fold = []
    gts, prs = [], []

    for fold_idx in range(6):
        cfg = os.path.join(args.cfg_root, args.cfg.format(fold_idx))

        # load config file
        if not cfg.endswith('.yaml'): cfg += '.yaml'
        assert os.path.isfile(cfg), "no config of {}".format(cfg)
        with open(cfg, encoding='UTF8') as cfg_file:
            cfg = yaml.safe_load(cfg_file)

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
        # model = MLPOnly(device)
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
        # compute METRIC
        prs_all_denorm = test_dataset.transform_output.denorm(prs_all)
        gts_all_denorm = test_dataset.transform_output.denorm(gts_all)
        # parsing into C-RATE
        crate_parser = ParserCRATE(cfg)
        prs_parsing = crate_parser.parsing(prs_all_denorm)
        gts_parsing = crate_parser.parsing(gts_all_denorm)
        # reshape data (40, 7)
        prs_reshape = []
        gts_reshape = []
        for crate in crate_parser.crates:
            prs_reshape.append(prs_parsing['specific_capacity'][crate].cpu().numpy())
            gts_reshape.append(gts_parsing['specific_capacity'][crate].cpu().numpy())
        prs_reshape = np.stack(prs_reshape, axis=1)
        gts_reshape = np.stack(gts_reshape, axis=1)
        prs.append(prs_reshape)
        gts.append(gts_reshape)
        # flatten data (280)
        pr_flat = prs_reshape.flatten()
        gt_flat = gts_reshape.flatten()
        # keep datas
        cell_ids = test_dataset.cell_ids
        prs_all_fold.append(pr_flat)
        gts_all_fold.append(gt_flat)
        ids_all_fold.append(cell_ids)

    prs_all_fold = np.concatenate(prs_all_fold)
    gts_all_fold = np.concatenate(gts_all_fold)
    r2_test = r2_score(gts_all_fold, prs_all_fold)
    # draw R2
    save_name = 'tmp_r2_ours.png'
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(gts_all_fold, prs_all_fold, s=15, c='white', edgecolors='blue')
    min_val = -20 # gts_all_fold.min()
    max_val = 205 # gts_all_fold.max()
    xs = np.arange(min_val, max_val, (max_val-min_val)/500)
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.scatter(xs, xs, s=0.15, c='black')
    plt.title("Test R2: {:.3f}".format(r2_test))
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.savefig(save_name, dpi=500)
    plt.clf() 
    print("... saved at", save_name)
    
    
    # save inference result as pickle
    import pickle
    save_dict = {}
    save_dict['gt'] = gts_all_fold
    save_dict['pr'] = prs_all_fold
    save_file = 'tmp_6fold_result_ours.pickle'
    with open(save_file, 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
    print('... saved at {}'.format(save_file))
    
    # print results
    cell_all = np.concatenate(ids_all_fold).transpose()
    prs = np.concatenate(prs).transpose()
    gts = np.concatenate(gts).transpose()
    # sort by cell-ids
    import pandas as pd
    prt_dict = {}
    prt_dict['cell_id'] = cell_all
    prt_dict['ids'] = np.arange(len(cell_all))
    prt_df = pd.DataFrame(prt_dict).sort_values('cell_id')
    ids = prt_df['ids'].to_numpy()
    gts = gts[:, ids]
    prs = prs[:, ids]
    
    print("---" * 20)
    print(str(list(cell_all[ids])).replace("[","").replace("]", "").replace("'",""))
    
    print("---" * 20)
    for i in gts:
        print(str(list(i)).replace("[","").replace("]", ""))
    print("---" * 20)
    for i in prs:
        print(str(list(i)).replace("[","").replace("]", ""))
    exit()


    print("---" * 20)
    for i in ids:
        data = gts[:, i]
        print(str(list(data)).replace("[","").replace("]", ""))
    print("---" * 20)
    for i in ids:
        data = prs[:, i]
        print(str(list(data)).replace("[","").replace("]", ""))