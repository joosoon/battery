import argparse
import os
from tqdm import tqdm
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pickle

import torch
from loader.batterycell import collate_batterycell, BatteryCellData
from model.custom import CUSTOM_FUSION

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
    

class ParserDesignParam():
    def __init__(self, cfg):
        self.params = [
                'cathode_loading_density',
                'cathode_porosity',
                'cathode_AM_thickness',
                'anode_loading_density',
                'anode_porosity',
                'anode_AM_thickness'
        ]
        self.cfg = cfg
        self.input_idx2param = {}
        for datas in self.cfg["DATASET"]["INPUT"]:
            name = datas['NAME']
            if name not in self.input_idx2param:
                self.input_idx2param[name] = {}
            items = datas['ITEM']
            for idx, item in enumerate(items):
                self.input_idx2param[name][idx] = item

    def parsing(self, data_dict):
        parse_dict = {}
        for k, v in data_dict.items():
            for data_idx in range(v.shape[1]):
                param = self.input_idx2param[k][data_idx]
                if param not in parse_dict:
                    parse_dict[param] = {}
                parse_dict[param] = v[:, data_idx]
        return parse_dict
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battery Performace')
    parser.add_argument('--csv_name', type=str, default="raw")
    parser.add_argument('--cfg', type=str, default="config/base.yaml")
    # param for dataset 
    parser.add_argument('--seed', type=int, default=7)
    # param for model
    parser.add_argument('--ckp_mode', type=str, default='latest', help="name/of/trained/weight")
    parser.add_argument('--ckp_file', type=str, default=None, help="path/to/trained/weight")
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cuda', choices=["cuda", "cpu"])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    # parma for visualization
    parser.add_argument('--vis_result', action='store_true')
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
    assert os.path.isfile(args.cfg)
    with open(args.cfg, encoding='UTF8') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        args.cfg = args.cfg[:-5]
    
    # set seed for reproduciblity
    seed = cfg["DATASET"].get("SEED", args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # set synthetic dataset
    for i, data in enumerate(cfg['DATASET']['INPUT']):
        cfg['DATASET']['INPUT'][i]['TYPE'] = args.csv_name
    for i, data in enumerate(cfg['DATASET']['OUTPUT']):
        cfg['DATASET']['OUTPUT'][i]['TYPE'] = args.csv_name

    # load dataset
    coll_fn = collate_batterycell
    test_dataset = BatteryCellData(cfg, "inference", seed)
    test_loader  = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=args.batch_size, 
                        num_workers=args.num_workers, shuffle=False, 
                        collate_fn=coll_fn)

    # load model
    model = CUSTOM_FUSION(cfg, device)
    if args.ckp_file is None:
        args.ckp_file = os.path.join(cfg["SAVE_ROOT"], "{}.pth".format(args.ckp_mode))
    ckp = torch.load(args.ckp_file, map_location=torch.device('cpu'))
    model.load_ckp(ckp['net'])
    
    # inference
    ins_all = {}
    prs_all = {}
    for idx, (data_input, data_gt) in enumerate(tqdm(test_loader)):
        # forward
        data_input = {k:v.to(device) for k, v in data_input.items()}
        data_gt = {k:v.to(device) for k, v in data_gt.items()}
        with torch.no_grad():
            outputs = model(data_input)
        # keep input
        for k in data_input.keys():
            input = data_input[k].clone().detach().cpu()
            if k not in ins_all:
                ins_all[k] = input
            else:
                ins_all[k] = torch.cat((ins_all[k], input))
        # keep output
        for k in outputs.keys():
            output = outputs[k].clone().detach().cpu()
            if k not in prs_all:
                prs_all[k] = output
            else:
                prs_all[k] = torch.cat((prs_all[k], output))

    # denorm and parsing input
    param_parser = ParserDesignParam(cfg)
    ins_all_denorm = test_dataset.transform_input.denorm(ins_all)
    ins_parsing = param_parser.parsing(ins_all_denorm)

    print("---" * 20)
    print("[input parsing]")
    for k, v in ins_parsing.items():
        print(k, v.shape)
            
    # denorm and parsing output
    crate_parser = ParserCRATE(cfg)
    prs_all_denorm = test_dataset.transform_output.denorm(prs_all)
    prs_parsing = crate_parser.parsing(prs_all_denorm)
    
    print("---" * 20)
    print("[output parsing]")
    for k, v in prs_parsing.items():
        print(k)
        for kk, vv in v.items():
            print("...", kk, vv.shape)
    
    # save results
    save_root = 'results_syn'
    save_dir = os.path.join(save_root, args.csv_name, 
                            os.path.basename(args.cfg).replace('.yaml', ''))
    os.makedirs(save_dir, exist_ok=True)
    
    save_dict = {}
    save_dict["input"] = ins_parsing
    save_dict["output"] = prs_parsing
    save_file = os.path.join(save_dir, 'parsing_result.pickle')
    print('... saving at {}'.format(save_file))
    with open(save_file, 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
        
    # draw results 
    if args.vis_result:
        print("... draw results")
        x_key = 'cathode_loading_density'
        y_key = 'cathode_porosity'
        for data_item, crates in prs_parsing.items():
            z_key = data_item
            save_dir_vis = os.path.join(save_dir, 'vis', data_item)
            os.makedirs(save_dir_vis, exist_ok=True)
            for crate in crates.keys():
                # 2d scatter
                xs = ins_parsing[x_key]
                ys = ins_parsing[y_key]
                zs = prs_parsing[z_key][crate]
                plt.scatter(xs, ys, c=zs)
                plt.xlabel(x_key)
                plt.ylabel(y_key)
                plt.title('{} - {}'.format(z_key, crate))
                plt.colorbar()
                save_name = os.path.join(save_dir_vis, "2D_{}.png".format(crate))
                plt.savefig(save_name, dpi=500)
                plt.clf()
                print("... saved at", save_name)
                
                # 3d scatter
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.scatter(xs, ys, zs, c=zs)
                plt.xlabel(x_key)
                plt.ylabel(y_key)
                plt.title('{} - {}'.format(z_key, crate))
                save_name = os.path.join(save_dir_vis, "3D_{}.png".format(crate))
                plt.savefig(save_name, dpi=500)
                plt.clf()
                print("... saved at", save_name)
            
            # 3D all together
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            for i, crate in enumerate(crates.keys()):
                z_key = data_item
                xs = ins_parsing[x_key]
                ys = ins_parsing[y_key]
                zs = prs_parsing[z_key][crate]
                ax.scatter(xs, ys, zs, label=crate)
            plt.legend(loc='upper left', bbox_to_anchor=(-0.17, 1.0),)
            plt.xlabel(x_key)
            plt.ylabel(y_key)
            plt.title('{} - all'.format(z_key))
            save_name = os.path.join(save_dir_vis, "3D_all.png")
            plt.savefig(save_name, dpi=500)
            plt.clf()
            print("... saved at", save_name)
                