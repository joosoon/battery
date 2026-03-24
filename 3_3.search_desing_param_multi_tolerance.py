import argparse
import os
from tqdm import tqdm
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pickle
import cv2

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
    


def normalize_to_int(arr, max_val=None, min_val=None, size=None):
    if max_val is None: max_val = arr.max()
    if min_val is None: min_val = arr.min()
    if size is None: size = len(np.unique(arr))-1
    arr_norm = np.around((arr - min_val) / (max_val - min_val) * size).astype(np.int64)
    return arr_norm

def check_search_result(target, result, query):
    # get range of targets
    cl_tar = target['cathode_loading_density'].numpy()
    cp_tar = target['cathode_porosity'].numpy()
    ct_tar = target['cathode_AM_thickness'].numpy()
    al_tar = target['anode_loading_density'].numpy()
    ap_tar = target['anode_porosity'].numpy()
    at_tar = target['anode_AM_thickness'].numpy()
    cl_min, cl_max = cl_tar.min(), cl_tar.max()
    cp_min, cp_max = cp_tar.min(), cp_tar.max()
    ct_min, ct_max = ct_tar.min(), ct_tar.max()
    al_min, al_max = al_tar.min(), al_tar.max()
    ap_min, ap_max = ap_tar.min(), ap_tar.max()
    at_min, at_max = at_tar.min(), at_tar.max()
    cl_nums = len(np.unique(cl_tar))-1
    cp_nums = len(np.unique(cp_tar))-1
    # ct_nums = len(np.unique(ct_tar))-1
    ct_nums = cp_nums
    al_nums = len(np.unique(al_tar))-1
    ap_nums = len(np.unique(ap_tar))-1
    # at_nums = len(np.unique(at_tar))-1
    at_nums = ap_nums
    nums = {}
    nums['cl'] = cl_nums
    nums['cp'] = cp_nums
    nums['ct'] = cp_nums
    nums['al'] = al_nums
    nums['ap'] = ap_nums
    nums['at'] = ap_nums
    # convert target (raw to normalized int)
    int_tar = {}
    int_tar['cl'] = normalize_to_int(cl_tar, cl_max, cl_min, cl_nums)
    int_tar['cp'] = normalize_to_int(cp_tar, cp_max, cp_min, cp_nums)
    int_tar['ct'] = normalize_to_int(ct_tar, ct_max, ct_min, ct_nums)
    int_tar['al'] = normalize_to_int(al_tar, al_max, al_min, al_nums)
    int_tar['ap'] = normalize_to_int(ap_tar, ap_max, ap_min, ap_nums)
    int_tar['at'] = normalize_to_int(at_tar, at_max, at_min, at_nums)
    # convert result (raw to normalized int) with same range
    cl_res = result['cathode_loading_density'].numpy()
    cp_res = result['cathode_porosity'].numpy()
    ct_res = result['cathode_AM_thickness'].numpy()
    al_res = result['anode_loading_density'].numpy()
    ap_res = result['anode_porosity'].numpy()
    at_res = result['anode_AM_thickness'].numpy()
    int_res = {}
    int_res['cl'] = normalize_to_int(cl_res, cl_max, cl_min, cl_nums)
    int_res['cp'] = normalize_to_int(cp_res, cp_max, cp_min, cp_nums)
    int_res['ct'] = normalize_to_int(ct_res, ct_max, ct_min, ct_nums)
    int_res['al'] = normalize_to_int(al_res, al_max, al_min, al_nums)
    int_res['ap'] = normalize_to_int(ap_res, ap_max, ap_min, ap_nums)
    int_res['at'] = normalize_to_int(at_res, at_max, at_min, at_nums)
    # convert query (raw to normalized int) with same range
    cl_query = query['cathode_loading_density'].numpy()
    cp_query = query['cathode_porosity'].numpy()
    ct_query = query['cathode_AM_thickness'].numpy()
    al_query = query['anode_loading_density'].numpy()
    ap_query = query['anode_porosity'].numpy()
    at_query = query['anode_AM_thickness'].numpy()
    int_que = {}
    int_que['cl'] = normalize_to_int(cl_query, cl_max, cl_min, cl_nums)
    int_que['cp'] = normalize_to_int(cp_query, cp_max, cp_min, cp_nums)
    int_que['ct'] = normalize_to_int(ct_query, ct_max, ct_min, ct_nums)
    int_que['al'] = normalize_to_int(al_query, al_max, al_min, al_nums)
    int_que['ap'] = normalize_to_int(ap_query, ap_max, ap_min, ap_nums)
    int_que['at'] = normalize_to_int(at_query, at_max, at_min, at_nums)

    # genrate empty array in range of target
    def check_with_arr(x, y, offset=3, vis=False):
        check_arr = np.zeros((nums[x]+1, nums[y]+1), dtype=np.uint8)
        check_arr[int_res[y], int_res[x]] = 100
        cnd = check_arr[
                        int_que[y]-offset:int_que[y]+offset,
                        int_que[x]-offset:int_que[x]+offset,
                        ]
        success = True if cnd.sum() > 0 else False
        if not vis:
            return success
        else:
            check_arr[int_que[y]-offset:int_que[y]+offset,
                      int_que[x]-offset:int_que[x]+offset,
                    ] = 255
            check_arr = np.flip(check_arr, 0)
            return success, check_arr
    # success, check_arr = check_with_arr(x='cl', y='cp', offset=3, vis=True)
    # cv2.imwrite("tmp_cnd_1.png", check_arr)
    # print(success)
    # success, check_arr = check_with_arr(x='al', y='ap', offset=3, vis=True)
    # cv2.imwrite("tmp_cnd_2.png", check_arr)
    # print(success)
    # success, check_arr = check_with_arr(x='al', y='cp', offset=3, vis=True)
    # cv2.imwrite("tmp_cnd_3.png", check_arr)
    # print(success)
    # success, check_arr = check_with_arr(x='cl', y='ap', offset=3, vis=True)
    # cv2.imwrite("tmp_cnd_4.png", check_arr)
    # print(success)
    # success, check_arr = check_with_arr(x='cl', y='al', offset=3, vis=True)
    # cv2.imwrite("tmp_cnd_5.png", check_arr)
    # print(success)
    # success, check_arr = check_with_arr(x='cp', y='ap', offset=3, vis=True)
    # cv2.imwrite("tmp_cnd_6.png", check_arr)
    # print(success)

    # check_arr = np.zeros((
    #                 nums['cl']+1, nums['cp']+1, 
    #                 nums['al']+1, nums['ap']+1
    #                 ))
    # check_arr[int_res['cl'], int_res['cp'], int_res['al'], int_res['ap']] = 100
    # offset = 3
    # cnd = check_arr[
    #                 int_que['cl']-offset:int_que['cl']+offset,
    #                 int_que['cp']-offset:int_que['cp']+offset,
    #                 # int_que['al']-offset:int_que['al']+offset,
    #                 # int_que['ap']-offset:int_que['ap']+offset,
    #                 ]


    # print(nums)
    
    check_arr = np.zeros((nums['cl']+1, nums['cp']+1, nums['ct']+1, 
                          nums['al']+1, nums['ap']+1, nums['at']+1))
    check_arr[int_res['cl'], int_res['cp'], int_res['ct'],
              int_res['al'], int_res['ap'], int_res['at']] = 100
    # offset = 2
    # cnd = check_arr[
    #                 int_que['cl']-offset:int_que['cl']+offset,
    #                 int_que['cp']-offset:int_que['cp']+offset,
    #                 int_que['ct']-offset:int_que['ct']+offset,
    #                 int_que['al']-offset:int_que['al']+offset,
    #                 int_que['ap']-offset:int_que['ap']+offset,
    #                 int_que['at']-offset:int_que['at']+offset,
    #                 ]
    cnd = check_arr[
                    int_que['cl']-2:int_que['cl']+2,
                    int_que['cp']-1:int_que['cp']+1,
                    int_que['ct']-1:int_que['ct']+1,
                    int_que['al']-4:int_que['al']+4,
                    int_que['ap']-1:int_que['ap']+1,
                    int_que['at']-1:int_que['at']+1,
                    ]
    success = True if cnd.sum() > 0 else False
    return success

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battery Performace')
    # param for target (syn data)
    parser.add_argument('--csv_name', type=str, default="raw")
    parser.add_argument('--cfg_model', type=str, default="config/base.yaml")
    # param for query (real data)
    parser.add_argument('--cfg', type=str, default="config/search_param_s.yaml")
    parser.add_argument('--tolerance', type=float, default=None)
    # param for dataset 
    parser.add_argument('--seed', type=int, default=7)
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cpu', choices=["cuda", "cpu"])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
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
    
    # load real data
    coll_fn = collate_batterycell
    test_dataset = BatteryCellData(cfg, "test", seed)
    test_loader  = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=args.batch_size, 
                        num_workers=args.num_workers, shuffle=False, 
                        collate_fn=coll_fn)
    ins_all = {}
    gts_all = {}
    cell_ids = []
    for idx, (data_input, data_gt) in enumerate(test_loader):
        cell_id = test_loader.dataset.cell_ids[idx]
        cell_ids.append(cell_id)
        # keep input
        for k in data_input.keys():
            input = data_input[k].clone().detach().cpu()
            if k not in ins_all:
                ins_all[k] = input
            else:
                ins_all[k] = torch.cat((ins_all[k], input))
        # keep gt
        for k in data_gt.keys():
            gt = data_gt[k].clone().detach().cpu()
            if k not in gts_all:
                gts_all[k] = gt
            else:
                gts_all[k] = torch.cat((gts_all[k], gt))
    # denorm and parsing input
    param_parser = ParserDesignParam(cfg)
    ins_all_denorm = test_dataset.transform_input.denorm(ins_all)
    ins_real = param_parser.parsing(ins_all_denorm)
    # denorm and parsing gt
    crate_parser = ParserCRATE(cfg)
    gts_all_denorm = test_dataset.transform_output.denorm(gts_all)
    gts_real = crate_parser.parsing(gts_all_denorm)
    
    # load synthetic data
    syn_file = os.path.join('results_syn', args.csv_name, 
                            os.path.basename(args.cfg_model).replace('.yaml', ''),
                            'parsing_result.pickle')
    with open(syn_file, 'rb') as f:
        mapping = pickle.load(f)
    ins_syn = mapping['input']
    prs_syn = mapping['output']
    
    
    print("---" * 20)
    print("[input real]")
    for k, v in ins_real.items():
        print(k, v.shape)
    print("[input syn]")
    for k, v in ins_syn.items():
        print(k, v.shape)
    
    print("---" * 20)
    print("[gt real]")
    for k, v in gts_real.items():
        print(k)
        for kk, vv in v.items():
            print("...", kk, vv.shape)
    print("[pred syn]")
    for k, v in prs_syn.items():
        print(k)
        for kk, vv in v.items():
            print("...", kk, vv.shape)
    
    tol2prt = {}
    tol_list = [0.00, 0.05, 0.10, 0.15, 0.20]
    tol_list = np.linspace(0.00, 0.25, 200)
    for tolerance in tqdm(tol_list):
        # find param with GT
        records = []
        records_crate = []
        records_select_ratio_all = []
        records_select_ratio_success = []
        for i, cell_id in enumerate(tqdm(cell_ids)):
            # get query (real data)
            query_inputs = {}
            for data_type, vals in ins_real.items():
                query_inputs[data_type] = vals[i]
            # get results over all c-rates
            for data_type, crates in gts_real.items():
                cnd_sum = None
                records_crate_cell = []
                for crate_i, crate in enumerate(crates.keys()):
                    vals = crates[crate]
                    query_val = vals[i]
                    targets = prs_syn[data_type][crate]
                    # find values of target larger than query
                    cnd = targets >= query_val * (1-tolerance)
                    cnd_sum = cnd if cnd_sum is None else torch.bitwise_and(cnd_sum, cnd)
                    # get success for each crate
                    result_inputs = {}
                    for k, v in ins_syn.items():
                        result_inputs[k] = v[cnd].clone()
                    success_crate = check_search_result(ins_syn, result_inputs, query_inputs)
                    records_crate_cell.append(success_crate)
                records_crate.append(records_crate_cell)
                # check search success
                result_inputs = {}
                for k, v in ins_syn.items():
                    result_inputs[k] = v[cnd_sum].clone()
                success = check_search_result(ins_syn, result_inputs, query_inputs)
                records.append(success)
                # keep select ratio
                select_ratio = cnd_sum.sum().item()/len(cnd_sum)
                records_select_ratio_all.append(select_ratio)
                if success: records_select_ratio_success.append(select_ratio)
        acc_all = sum(records) / len(records) * 100
        avg_select_ratio_all = 100*np.mean(records_select_ratio_all)
        avg_select_ratio_suc = 100*np.mean(records_select_ratio_success)
        # print for recodring
        prt_str = "{},{:.2f},".format(tolerance, acc_all)
        print(tolerance, end=',')
        print(acc_all, end=',')
        crates = ['0.1C','0.2C','0.5C','1C','2C','3C','5C']
        records_crate = np.array(records_crate)
        for i, crate in enumerate(crates):
            record = records_crate[:, i]
            acc = record.sum() / len(record) * 100
            prt_str += "{:.2f},".format(acc)
            print(acc, end=',')
        prt_str += "{:.2f},{:.2f}".format(avg_select_ratio_all, avg_select_ratio_suc)
        print(avg_select_ratio_all, end=',')
        print(avg_select_ratio_suc)
        tol2prt[tolerance] = prt_str
    print("---" * 20)
    for k, v in tol2prt.items():
        print(v)