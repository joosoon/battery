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

    check_arr = np.zeros((nums['cl']+1, nums['cp']+1, nums['ct']+1, 
                          nums['al']+1, nums['ap']+1, nums['at']+1))
    check_arr[int_res['cl'], int_res['cp'], int_res['ct'],
              int_res['al'], int_res['ap'], int_res['at']] = 100
    offset = 3
    cnd = check_arr[
                    int_que['cl']-offset:int_que['cl']+offset,
                    int_que['cp']-offset:int_que['cp']+offset,
                    int_que['ct']-offset:int_que['ct']+offset,
                    int_que['al']-offset:int_que['al']+offset,
                    int_que['ap']-offset:int_que['ap']+offset,
                    int_que['at']-offset:int_que['at']+offset,
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
    
    
    # find param with GT
    print("---" * 20)
    records = []
    records_crate = []
    tolerance = 0.05
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
                print(">>> CRATE {:<4s} | Val {:.3f} | Res {:.2f}%({}/{}) | Sum {:.2f}%({}/{})".format(
                      crate, query_val.item(), 
                      100*cnd.sum().item()/len(cnd), cnd.sum().item(), len(cnd),
                      100*cnd_sum.sum().item()/len(cnd_sum), cnd_sum.sum().item(), len(cnd_sum)
                      ))

                # get success for each crate
                result_inputs = {}
                for k, v in ins_syn.items():
                    result_inputs[k] = v[cnd].clone()
                success_crate = check_search_result(ins_syn, result_inputs, query_inputs)
                records_crate_cell.append(success_crate)
                print(success_crate)

            print("---" * 20)
            exit()


            
            # draw     
                # # get success for each crate
                # result_inputs = {}

                # fig = plt.figure(figsize=(20, 10))
                # col_i = 0
                # for k, v in ins_syn.items():
                #     result_inputs[k] = v[cnd].clone()
                #     targets = v
                #     results = v[cnd]
                #     query = query_inputs[k]

                #     print("... draw input {} - {}".format(crate, k))
                #     # tmp: draw each design parameter
                #     import matplotlib.pyplot as plt
                #     col_i += 1
                #     plt.subplot(2, 3, col_i)                    
                #     # draw targets 
                #     xs = targets
                #     ys = [1 for _ in range(len(xs))]                
                #     plt.scatter(xs, ys, label='targets')
                #     # draw results 
                #     xs = list(results)
                #     ys = [2 for _ in range(len(xs))]                    
                #     # tmp_num = 30
                #     # tmp_xs = [min(xs) for _ in range(tmp_num)] + \
                #     #          [max(xs) for _ in range(tmp_num)]   
                #     # tmp_ys = [2-(_*1/tmp_num) for _ in range(tmp_num)] + \
                #     #          [2-(_*1/tmp_num) for _ in range(tmp_num)]
                #     # xs += tmp_xs
                #     # ys += tmp_ys
                #     plt.scatter(xs, ys, label='results')                    
                #     # draw gt 
                #     xs = [query]
                #     ys = [3 for _ in range(len(xs))]                   
                #     tmp_num = 60
                #     tmp_xs = [min(xs) for _ in range(tmp_num)]   
                #     tmp_ys = [3-(_*2/tmp_num) for _ in range(tmp_num)]
                #     xs += tmp_xs
                #     ys += tmp_ys
                #     plt.scatter(xs, ys, label='GT({:.2f})'.format(query))
                #     plt.title("{} - {}".format(crate, k))
                #     plt.legend(loc='best')
                # plt.suptitle("C-rate {} ({:.3f})".format(crate, query_val))
                # save_name = "tmp_search_{}_{}.png".format(cell_id, crate)
                # plt.savefig(save_name, dpi=500)
                # plt.clf()
                # print("... saved at", save_name)
            exit()
                # success_crate = check_search_result(ins_syn, result_inputs, query_inputs)
                # records_crate_cell.append(success_crate)
            records_crate.append(records_crate_cell)
            # check search success
            result_inputs = {}
            for k, v in ins_syn.items():
                result_inputs[k] = v[cnd_sum].clone()
            success = check_search_result(ins_syn, result_inputs, query_inputs)
            records.append(success)
            
            if args.vis_result:
                # draw cathode
                x_key = 'cathode_loading_density'
                y_key = 'cathode_porosity'
                for data_type, crates in gts_real.items():
                    cnd_sum = None
                    fig = plt.figure(figsize=(20, 10))
                    for crate_i, crate in enumerate(crates.keys()):
                        vals = crates[crate]
                        query_val = vals[i]
                        targets = prs_syn[data_type][crate]
                        # find values of target larger than query
                        cnd = targets >= query_val * (1-tolerance)
                        cnd_sum = cnd if cnd_sum is None else torch.bitwise_and(cnd_sum, cnd)
                        print("... draw {}".format(crate))
                        # draw 2D scatter
                        z_key = data_type
                        plt.subplot(2, 4, crate_i+1)
                        # all synthetic data
                        xs = ins_syn[x_key]
                        ys = ins_syn[y_key]
                        zs = prs_syn[z_key][crate]
                        plt.scatter(xs, ys, c=zs)
                        plt.colorbar()
                        # search result data
                        xs = ins_syn[x_key][cnd]
                        ys = ins_syn[y_key][cnd]
                        zs = prs_syn[z_key][crate][cnd]
                        plt.scatter(xs, ys, c='white', alpha=0.1)
                        plt.scatter([], [], c='gray', alpha=0.3, label='results')
                        # query input
                        xs = [query_inputs[x_key]]
                        ys = [query_inputs[y_key]]
                        plt.scatter(xs, ys, c='red', alpha=1, label='query')
                        plt.xlabel(x_key)
                        plt.ylabel(y_key)
                        plt.legend(loc='best')
                        plt.title("C-rate {} | query {:.2f}".format(crate, query_val))
                    print("... draw all")
                    # draw final results
                    plt.subplot(2, 4, crate_i+2)
                    # all synthetic data
                    xs = ins_syn[x_key]
                    ys = ins_syn[y_key]
                    zs = prs_syn[z_key][crate]
                    plt.scatter(xs, ys, c='blue', alpha=0.1)
                    # sum of search result
                    xs = ins_syn[x_key][cnd_sum]
                    ys = ins_syn[y_key][cnd_sum]
                    plt.scatter(xs, ys, c='gray', label='results')
                    # query input
                    xs = [query_inputs[x_key]]
                    ys = [query_inputs[y_key]]
                    plt.scatter(xs, ys, c='red', alpha=1, label='query')
                    plt.xlabel(x_key)
                    plt.ylabel(y_key)
                    plt.legend(loc='best')
                    plt.title("FINAL")
                    plt.suptitle("{} - query cell {}".format(data_type, cell_id), fontsize=30)
                    save_dir = os.path.join('results_syn', args.csv_name, 
                                            os.path.basename(args.cfg_model).replace('.yaml', ''), 
                                            'vis', 'search')
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = os.path.join(save_dir, '{}_{}_cathode.png'.format(data_type, cell_id))
                    plt.savefig(save_name, dpi=500)
                    plt.clf()
                    print("saved at", save_name)

                # draw anode
                x_key = 'anode_loading_density'
                y_key = 'anode_porosity'
                for data_type, crates in gts_real.items():
                    cnd_sum = None
                    fig = plt.figure(figsize=(20, 10))
                    for crate_i, crate in enumerate(crates.keys()):
                        vals = crates[crate]
                        query_val = vals[i]
                        targets = prs_syn[data_type][crate]
                        # find values of target larger than query
                        cnd = targets >= query_val * (1-tolerance)
                        cnd_sum = cnd if cnd_sum is None else torch.bitwise_and(cnd_sum, cnd)
                        print("... draw {}".format(crate))
                        # draw 2D scatter
                        z_key = data_type
                        plt.subplot(2, 4, crate_i+1)
                        # all synthetic data
                        xs = ins_syn[x_key]
                        ys = ins_syn[y_key]
                        zs = prs_syn[z_key][crate]
                        plt.scatter(xs, ys, c=zs)
                        plt.colorbar()
                        # search result data
                        xs = ins_syn[x_key][cnd]
                        ys = ins_syn[y_key][cnd]
                        zs = prs_syn[z_key][crate][cnd]
                        plt.scatter(xs, ys, c='white', alpha=0.1)
                        plt.scatter([], [], c='gray', alpha=0.3, label='results')
                        # query input
                        xs = [query_inputs[x_key]]
                        ys = [query_inputs[y_key]]
                        plt.scatter(xs, ys, c='red', alpha=1, label='query')
                        plt.xlabel(x_key)
                        plt.ylabel(y_key)
                        plt.legend(loc='best')
                        plt.title("C-rate {} | query {:.2f}".format(crate, query_val))
                    print("... draw all")
                    # draw final results
                    plt.subplot(2, 4, crate_i+2)
                    # all synthetic data
                    xs = ins_syn[x_key]
                    ys = ins_syn[y_key]
                    zs = prs_syn[z_key][crate]
                    plt.scatter(xs, ys, c='blue', alpha=0.1)
                    # sum of search result
                    xs = ins_syn[x_key][cnd_sum]
                    ys = ins_syn[y_key][cnd_sum]
                    plt.scatter(xs, ys, c='gray', label='results')
                    # query input
                    xs = [query_inputs[x_key]]
                    ys = [query_inputs[y_key]]
                    plt.scatter(xs, ys, c='red', alpha=1, label='query')
                    plt.xlabel(x_key)
                    plt.ylabel(y_key)
                    plt.legend(loc='best')
                    plt.title("FINAL")
                    plt.suptitle("{} - query cell {}".format(data_type, cell_id), fontsize=30)
                    save_dir = os.path.join('results_syn', args.csv_name, 
                                            os.path.basename(args.cfg_model).replace('.yaml', ''), 
                                            'vis', 'search')
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = os.path.join(save_dir, '{}_{}_anode.png'.format(data_type, cell_id))
                    plt.savefig(save_name, dpi=500)
                    plt.clf()
                    print("saved at", save_name)
    
    print("---" * 20)
    acc = sum(records) / len(records) * 100
    print("ACCURACY: {:.2f}% ({}/{})".format(acc, sum(records), len(records)))
    
    crates = ['0.1C','0.2C','0.5C','1C','2C','3C','5C']
    records_crate = np.array(records_crate)
    for i, crate in enumerate(crates):
        record = records_crate[:, i]
        acc = record.sum() / len(record) * 100
        print("... [{}] {:.2f}% ({}/{})".format(crate, acc, record.sum(), len(record)))

    for record in records_crate:
        for i, val in enumerate(record):
            end_str = ',' if i < len(record)-1 else "\n"
            val = 1 if val == True else 0
            print(val, end=end_str)
                

        