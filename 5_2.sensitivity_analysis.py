import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import yaml
import warnings 
warnings.filterwarnings('ignore')
# for deep learning model
import torch
from loader.batterycell import collate_batterycell, BatteryCellData
from model.custom import CUSTOM_FUSION
# for sensitivity analysis
from SALib.sample import saltelli
from SALib.analyze import sobol


def init_env(args):
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

    return device, seed, cfg

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

class SynDesignParam():
    def __init__(self, samples):
        self.param = {}
        for target, val in samples.items():
            self.param[target] = np.array(val)
        self.num_data = len(val)
        # calculate cathode param
        self.param['cathode_Electrode weight (w/o foil,g)'] = self.param['cathode_loading_density'] / 1000 * 0.25 * math.pi
        self.param['cathode_Weight (mg)'] = self.param['cathode_Electrode weight (w/o foil,g)'] * 1000 + 6.4
        self.param['cathode_Loading mass of AM (mg)'] = self.param['cathode_Weight (mg)']-6.4
        self.param['cathode_Loading mass of AM (mg)'] *= 0.93
        self.param['cathode_Loading mass of AM (mg)'] = np.round(self.param['cathode_Loading mass of AM (mg)'], 2)
        self.param['cathode_Loading density (mAh/cm2)'] = self.param['cathode_Loading mass of AM (mg)'] * 210 / 0.25 / math.pi / 1000
        self.param['cathode_Theoretical capacity (mAh)'] = self.param['cathode_Loading mass of AM (mg)']*210
        self.param['cathode_Theoretical capacity (mAh)'] = np.round(self.param['cathode_Theoretical capacity (mAh)'], -1)
        self.param['cathode_AM_thickness']  = self.param['cathode_Electrode weight (w/o foil,g)'].copy()
        self.param['cathode_AM_thickness'] /= 1 - self.param['cathode_porosity'] / 100
        self.param['cathode_AM_thickness'] *= 10000 / math.pi / 0.25
        self.param['cathode_AM_thickness'] *= (0.93/4.7) + (0.04/1.9) + (0.03/1.78)
        self.param['cathode_전극 두께 (㎛)'] = self.param['cathode_AM_thickness'] + 30
        # calculate anode param
        self.param['anode_Theoretical capacity (mAh)'] = self.param['cathode_Theoretical capacity (mAh)'].copy()
        self.param['anode_Theoretical capacity (mAh)'] *= self.param['np_ratio'] * 196 / 100
        self.param['anode_Theoretical capacity (mAh)'] = np.round(self.param['anode_Theoretical capacity (mAh)'], -1)
        self.param['anode_Loading mass of AM (mg)'] = self.param['anode_Theoretical capacity (mAh)'] / 360
        self.param['anode_Loading mass of AM (mg)'] = np.round(self.param['anode_Loading mass of AM (mg)'], 2)
        self.param['anode_Weight (mg)'] = self.param['anode_Loading mass of AM (mg)'] / 0.96
        self.param['anode_Weight (mg)'] += 45
        self.param['anode_Weight (mg)'] = np.round(self.param['anode_Weight (mg)'], 1)
        self.param['anode_Electrode weight (w/o foil,g)'] = self.param['anode_Weight (mg)'] - 45
        self.param['anode_Electrode weight (w/o foil,g)'] /= 1000
        self.param['anode_loading_density'] = self.param['anode_Electrode weight (w/o foil,g)'] * 1000 / 0.49 / math.pi
        self.param['anode_AM_thickness']  = self.param['anode_Electrode weight (w/o foil,g)'].copy()
        self.param['anode_AM_thickness'] /= 1 - self.param['anode_porosity']
        self.param['anode_AM_thickness'] *= 10000 / math.pi / 0.49
        self.param['anode_AM_thickness'] *= (0.96/2.26) + (0.01/1.95) + (0.015/1.254)

# param_sample = {}
# param_sample['cathode_loading_density'] = [9.80394449446075,9.80394449446075,10.0585924034078,9.93126844893427,9.80394449446075,9.93126844893427,9.93126844893427,10.0585924034078,9.93126844893427,9.80394449446075,10.0585924034078,9.93126844893427,10.0585924034078,10.3132403123548,10.5678882213018]
# param_sample['cathode_porosity']        = [0.254334421434564,0.254334421434564,0.234966484328968,0.244650452881766,0.254334421434564,0.244650452881766,0.232267673420811,0.222424951285181,0.206242848791008,0.277636470764734,0.234966484328968,0.206242848791008,0.242299073936038,0.192144683186862,0.244943906474275]
# param_sample['anode_porosity']          = [0.280684345126312,0.272378844416441,0.272378844416441,0.279054267862162,0.272378844416441,0.285729691307883,0.277326511205622,0.277326511205622,0.272378844416441,0.257064637688023,0.263818595527223,0.248220169089071,0.283976103250398,0.309874839941341,0.256057740303739]
# param_sample['np_ratio']                = [1.22210884353742,1.28231292517007,1.24900609594487,1.25201396348013,1.28231292517007,1.24194414607948,1.24194414607948,1.22581500132521,1.26544038668099,1.29251700680272,1.24900609594487,1.27551020408163,1.2953882851842,1.26259364505296,1.23141849332326]
# param_syn = SynDesignParam(param_sample)
# key = 'anode_loading_density'
# datas = param_syn.param[key]
# for data in datas:
#     print(data)
# exit()        



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
    args = parser.parse_args()
    
    # initialization
    device, seed, cfg = init_env(args)
    
    # load model
    model = CUSTOM_FUSION(cfg, device)
    if args.ckp_file is None:
        args.ckp_file = os.path.join(cfg["SAVE_ROOT"], "{}.pth".format(args.ckp_mode))
    ckp = torch.load(args.ckp_file, map_location=torch.device('cpu'))
    model.load_ckp(ckp['net'])
    
    # modify config
    cfg['DATASET']['NUM_TEST'] = 0
    for in_out in ["INPUT", "OUTPUT"]:
        for i in range(len(cfg['DATASET'][in_out])):
            cfg['DATASET'][in_out][i]['TYPE'] = 'sobol'


    # setting for data sampling
    param2range = {}
    param2range['cathode_loading_density']  = {"min":  9.5, 'max': 17.0, 'mean': 13.4}
    param2range['cathode_porosity']         = {"min": 16.5, 'max': 39.0, 'mean': 28.5}
    param2range['cathode_AM_thickness']     = {"min": 29.0, 'max': 53.5, 'mean': 40.0}
    param2range['anode_loading_density']    = {"min":  6.0, 'max': 11.0, 'mean':  8.5}
    param2range['anode_porosity']           = {"min": 0.20, 'max': 0.45, 'mean': 0.33}
    param2range['anode_AM_thickness']       = {"min": 41.5, 'max': 75.5, 'mean': 55.8}
    
    num_div = 1
    param_group = [
                   'cathode_loading_density',
                   'cathode_porosity',
                   'cathode_AM_thickness',
                   'anode_loading_density',
                   'anode_porosity',
                   'anode_AM_thickness',
                   ]

    # divide ranges
    param2ranges = {}
    start2end = {}
    for param in param_group:
        min_val = param2range[param]["min"]
        max_val = param2range[param]["max"]
        ranges = np.linspace(min_val, max_val, num_div+1)
        start2end[param] = {}
        for i in range(len(ranges)-1):
            s = ranges[i]
            e = ranges[i+1]
            start2end[param][s] = e
        param2ranges[param] = ranges[:-1]
        print("[{}]:".format(param), ranges)

    
    # generate combinations
    ranges_all = []
    for target, ranges in param2ranges.items():
        ranges_all.append(ranges)
    combs = np.array(np.meshgrid(*ranges_all)).T.reshape(-1, len(ranges_all))
    print("... generate {} combinations".format(len(combs)))


    # comb2si = {}
    # comb2si[0]: {'cathode_loading_density': 1.05012, 'cathode_porosity': 0.96071, 'cathode_AM_thickness': 1.02137}
    # comb2si[1]: {'cathode_loading_density': 1.05012, 'cathode_porosity': 0.96071, 'cathode_AM_thickness': 1.02137}
    # comb2si[2]: {'cathode_loading_density': 1.05012, 'cathode_porosity': 0.96071, 'cathode_AM_thickness': 1.02137}
    # comb2si[3]: {'cathode_loading_density': 1.05012, 'cathode_porosity': 0.96071, 'cathode_AM_thickness': 1.02137}
    # comb2si[4]: {'cathode_loading_density': 1.05164, 'cathode_porosity': 0.95655, 'cathode_AM_thickness': 1.02533}
    # comb2si[5]: {'cathode_loading_density': 1.05164, 'cathode_porosity': 0.95655, 'cathode_AM_thickness': 1.02533}
    # comb2si[6]: {'cathode_loading_density': 1.05164, 'cathode_porosity': 0.95655, 'cathode_AM_thickness': 1.02533}
    # comb2si[7]: {'cathode_loading_density': 1.05164, 'cathode_porosity': 0.95655, 'cathode_AM_thickness': 1.02533}
    
    # # draw 3D heatmap
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.scatter(xs, ys, zs, c=color_target, edgecolors='none', alpha=0.005)
    
    
    # calculate Sobol index
    comb2si = {}
    for comb_i, comb in enumerate(combs):
        param_bounds = []
        for i, param in enumerate(param_group):
            s = comb[i]
            e = start2end[param][s]
            param_bounds.append([s, e])
        # generate data
        problem = {
        'num_vars': len(param_group),
        'names': param_group,
        'bounds': param_bounds}    
        param_values = saltelli.sample(problem, 1024)
        # save as csv
        save_root = 'dataset/csv'
        save_name = os.path.join(save_root, "sobol.csv")
        save_dict = {}
        save_dict['cell_id'] = np.arange(len(param_values))+1
        for param in param2range.keys():
            if param in param_group: 
                save_dict[param] = param_values[:, i]
            else:
                mean_val = param2range[param]['mean']
                save_dict[param] = np.full(len(param_values), fill_value=mean_val)
        for crate in [0.1,0.2,0.5,1,2,3,5]:
            save_dict['specific_capacity_{}C'.format(crate)] = np.ones(len(param_values))
        save_df = pd.DataFrame(save_dict)
        save_df.to_csv(save_name)
        # print("... saved {} datas at {}".format(len(save_df), save_name))
        # generate dataset and dataloader
        coll_fn = collate_batterycell
        test_dataset = BatteryCellData(cfg, 'train', seed)
        test_loader  = torch.utils.data.DataLoader(
                            dataset=test_dataset, batch_size=512, 
                            num_workers=4, shuffle=False, 
                            collate_fn=coll_fn)
        # generate outputs
        prs_all = {}
        gts_all = {}
        # for idx, (data_input, data_gt) in enumerate(tqdm(test_loader)):
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
        # de-normalization
        prs_all_denorm = test_dataset.transform_output.denorm(prs_all)
        # parsing into C-RATE
        crate_parser = ParserCRATE(cfg)
        prs_parsing = crate_parser.parsing(prs_all_denorm)
        # calculate Sobol Index
        data_item = 'specific_capacity'
        crate = '0.1C'
        outputs = prs_parsing[data_item][crate].cpu().numpy()
        Si = sobol.analyze(problem, outputs)
        s1 = Si['S1']
        st = Si['ST']
        sobol_result = {}
        for param, st in zip(param_group, Si['ST']):
            sobol_result[param] = np.round(st, 5)
        comb2si[comb_i] = sobol_result
        print("{:03d}".format(comb_i), sobol_result)
        
        data_item = 'specific_capacity'
        for s_1_t in ['S1', 'ST']:
            print("---" * 20)
            print("Sobol Index: {}".format(s_1_t))
            for crate in crate_parser.crates:
                outputs = prs_parsing[data_item][crate].cpu().numpy()
                Si = sobol.analyze(problem, outputs)
                sobol_result = {}
                for param, st in zip(param_group, Si[s_1_t]):
                    sobol_result[param] = np.round(st, 5)
                print("{:<4s}".format(crate), sobol_result)
