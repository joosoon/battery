import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .const import *
import natsort
from tqdm import tqdm
import copy 

import torch
from torch.utils.data import Dataset
import copy


class TransformCell():
    def __init__(self, cfg):
        self.norm_vals = {}
        self.norm_vals['cathode_loading_density'] = 13.000
        self.norm_vals['cathode_porosity'] = 25.000
        self.norm_vals['cathode_AM_thickness'] = 40.000
        self.norm_vals['anode_loading_density'] = 8.500
        self.norm_vals['anode_porosity'] = 0.300
        self.norm_vals['anode_AM_thickness'] = 56.000
        self.norm_vals['energy_0.1C'] = 0.005
        self.norm_vals['energy_0.2C'] = 0.005
        self.norm_vals['energy_0.5C'] = 0.005
        self.norm_vals['energy_1C'] = 0.004
        self.norm_vals['energy_2C'] = 0.003
        self.norm_vals['energy_3C'] = 0.002
        self.norm_vals['energy_5C'] = 0.001
        self.norm_vals['energy_density_0.1C'] = 0.500
        self.norm_vals['energy_density_0.2C'] = 0.500
        self.norm_vals['energy_density_0.5C'] = 0.500
        self.norm_vals['energy_density_1C'] = 0.400
        self.norm_vals['energy_density_2C'] = 0.300
        self.norm_vals['energy_density_3C'] = 0.230
        self.norm_vals['energy_density_5C'] = 0.120
        self.norm_vals['capacity_0.1C'] = 1.500
        self.norm_vals['capacity_0.2C'] = 1.400
        self.norm_vals['capacity_0.5C'] = 1.200
        self.norm_vals['capacity_1C'] = 1.000
        self.norm_vals['capacity_2C'] = 0.800
        self.norm_vals['capacity_3C'] = 0.600
        self.norm_vals['capacity_5C'] = 0.340
        self.norm_vals['specific_capacity_0.1C'] = 160.000
        self.norm_vals['specific_capacity_0.2C'] = 150.000
        self.norm_vals['specific_capacity_0.5C'] = 130.000
        self.norm_vals['specific_capacity_1C'] = 110.000
        self.norm_vals['specific_capacity_2C'] = 85.000
        self.norm_vals['specific_capacity_3C'] = 65.000
        self.norm_vals['specific_capacity_5C'] = 40.000
        
        self.norm_vals['charge_voltage_cycle0_len500'] = 1.0
        self.norm_vals['discharge_voltage_cycle0_len500'] = 1.0
        
        self.set_name2val(cfg)
        
    def set_name2val(self, cfg):
        self.name2val = {}
        for data in cfg:
            name = data['NAME']
            items = data['ITEM']
            self.name2val[name] = []
            for item in items:
                val = self.norm_vals[item]
                self.name2val[name].append(val)
            self.name2val[name] = np.array(self.name2val[name])
        
    def norm(self, data_dict):
        for k in data_dict.keys():
            norm_vals = self.name2val[k]
            if norm_vals is not None:
                data_dict[k] /= norm_vals
        return data_dict

    def denorm(self, data_dict):
        new_dict = copy.deepcopy(data_dict)
        for k in data_dict.keys():
            norm_vals = self.name2val[k]
            if norm_vals is not None:
                new_dict[k] *= norm_vals
        return new_dict



def collate_batterycell(batch):
    inputs, outputs = {}, {}
    # gather all data    
    for input, output in batch:
        for k, v in input.items():
            if k not in inputs: inputs[k] = []
            inputs[k].append(v)
        for k, v in output.items():
            if k not in outputs: outputs[k] = []
            outputs[k].append(v)
    # convert numpy to torch
    for k in inputs.keys():
        inputs[k] = np.stack(inputs[k]).astype(np.float32)
        inputs[k] = torch.from_numpy(inputs[k])
    for k in outputs.keys():
        outputs[k] = np.stack(outputs[k]).astype(np.float32) 
        outputs[k] = torch.from_numpy(outputs[k])   
    return inputs, outputs


# fixed testset
TEST_IDS = [
    '1.1-15', '1.1-16', '1.1-26', '1.1-3', '1.1-6', '1.2-12', '1.2-13', '1.2-2', '1.2-5', '1.3-14', 
    '1.3-28', '1.3-31', '1.3-6', '2.1-14', '2.1-15', '2.1-19', '2.1-20', '2.1-22', '2.1-24', '2.1-28', 
    '2.2-10', '2.2-13', '2.2-27', '2.2-32', '2.2-33', '2.3-11', '2.3-12', '2.3-15', '2.3-16', '2.3-17', 
    '2.3-21', '2.3-29', '2.3-32', '2.3-7', '3.1-1', '3.1-13', '3.1-16', '3.1-3', '3.2-12', '3.2-17', 
    '3.2-19', '3.2-20', '3.2-24', '3.2-4', '3.2-5', '3.2-8', '3.3-10', '3.3-21', '3.3-6', '3.3-8'
    ]

TEST_IDS = [ # total 240 cell (remove 6 cell) / test 50 cell / seed 5
    '1.1-12', '1.1-16', '1.1-4', '1.1-8', '1.2-12', '1.2-13', '1.2-15', '1.2-23', '1.2-27', '1.2-3', 
    '1.2-33', '1.2-5', '1.2-8', '1.3-1', '1.3-11', '1.3-12', '1.3-17', '1.3-18', '1.3-30', '1.3-4', 
    '2.1-1', '2.1-14', '2.1-15', '2.1-22', '2.1-24', '2.1-29', '2.1-31', '2.2-18', '2.2-20', '2.2-8', 
    '2.3-16', '2.3-19', '2.3-29', '2.3-3', '2.3-7', '3.1-14', '3.1-15', '3.1-17', '3.1-23', '3.1-3', 
    '3.2-14', '3.2-23', '3.2-6', '3.3-11', '3.3-12', '3.3-14', '3.3-16', '3.3-22', '3.3-24', '3.3-7']
TEST_IDS = [ # total 240 cell (remove 6 cell) / test 40 cell / seed 5
    '1.1-12', '1.1-16', '1.1-4', '1.1-8', '1.2-12', '1.2-13', '1.2-15', '1.2-23', '1.2-27', '1.2-3', 
    '1.2-33', '1.2-5', '1.2-8', '1.3-1', '1.3-11', '1.3-12', '1.3-17', '1.3-18', '1.3-30', '1.3-4', 
    '2.1-1', '2.1-14', '2.1-15', '2.1-22', '2.1-24', '2.1-29', '2.1-31', '2.2-18', '2.2-20', '2.2-8', 
    '2.3-16', '2.3-19', '2.3-29', '2.3-3', '2.3-7', '3.1-14', '3.1-15', '3.1-17', '3.1-23', '3.1-3', ]

# TEST_IDS = [ # original
#     '1.3-14', '3.2-12', '2.1-14', '2.3-11', '2.3-16', '1.1-16', '2.1-20', '3.2-4', '3.3-6', '3.1-3', 
#     '2.2-10', '2.3-12', '2.2-33', '1.2-12', '2.1-15', '3.3-8', '2.3-21', '2.3-17', '2.1-24', '2.1-22', 
#     '3.2-8', '2.1-19', '3.1-1', '2.3-32', '1.2-2', '1.1-26', '1.1-6', '3.2-5', '3.2-17', '1.3-6', 
#     '2.1-28', '2.3-29', '2.3-15', '1.2-5', '3.2-19', '3.1-16', '3.3-10', '1.1-15', '3.1-13', '1.3-28',
#     '1.2-13', '3.2-20', '2.3-7', '2.2-32', '1.1-3', '2.2-13', '3.3-21', '1.3-31', '2.2-27', '3.2-24'
#     ]
class BatteryCellData(Dataset):
    def __init__(self, cfg, mode="train", seed=777):
        assert mode in ["train", "test", "inference"]
        self.mode = mode

        self.cfg = copy.deepcopy(cfg)
        self.data_root = self.cfg["DATASET"]["ROOT"]

        # train-test split
        cell_ids = []
        for datas in self.cfg["DATASET"]["INPUT"]+self.cfg["DATASET"]["OUTPUT"]:
            if datas['DATA'] == 'csv':
                data_file = os.path.join(cfg["DATASET"]["ROOT"], datas["DATA"], datas["TYPE"]+'.csv')
                tmp_ids = sorted(pd.read_csv(data_file)['cell_id'].to_list())
                cell_ids = list(set(cell_ids+tmp_ids))
                # print("... {} - {} cell".format(datas['NAME'], len(tmp_ids)))
        # add smote dataset
        if cfg["DATASET"].get("TRAINSMOTE", None) is not None and mode=='train':
            smote_ids = []
            if isinstance(cfg["DATASET"]["TRAINSMOTE"], str):
                cfg["DATASET"]["TRAINSMOTE"] = [cfg["DATASET"]["TRAINSMOTE"]]
            for smote_file in cfg["DATASET"]["TRAINSMOTE"]:
                smote_ids += sorted(pd.read_csv(smote_file)['cell_id'].to_list())
            self.smote_ids = smote_ids
            cell_ids += smote_ids

        # set seed for reproduciblity
        np.random.seed(seed)
        torch.manual_seed(seed)
       
        # ###################################
        # # 1. shuffle and split TRAIN-TEST #
        # ###################################
        # cell_ids = sorted(cell_ids)
        # np.random.shuffle(cell_ids)
        # num_test = cfg["DATASET"]["NUM_TEST"]
        # num_test = int(num_test*len(cell_ids)) if num_test <= 1 else int(num_test)
        # if mode == 'test':
        #     self.cell_ids = cell_ids[:num_test]
        # elif mode == 'train':
        #     self.cell_ids = cell_ids[num_test:]
        # else:
        #     self.cell_ids = cell_ids
        # #################################
        # # 2. use fixed train-test split #
        # #################################
        # cell_ids = sorted(cell_ids)
        # np.random.shuffle(cell_ids)
        # if mode == 'test':
        #     self.cell_ids = TEST_IDS
        # elif mode == 'train':
        #     self.cell_ids = [cell_id for cell_id in cell_ids if cell_id not in TEST_IDS]
        # else:
        #     self.cell_ids = cell_ids
        ##################################
        # 3. pre-split testset in config #
        ##################################
        if cfg["DATASET"].get("TESTSET", None) is not None:
            TEST_IDS = TESTSET_BUFF[cfg["DATASET"]["TESTSET"]]
            
            # add validation set
            if mode == 'train' and cfg["DATASET"].get("VALSET", None) is not None:
                TEST_IDS += TESTSET_BUFF[cfg["DATASET"]["VALSET"]]
                
            if mode == 'test':
                self.cell_ids = TEST_IDS
            elif mode == 'train':
                self.cell_ids = [cell_id for cell_id in cell_ids if cell_id not in TEST_IDS]
            else:
                self.cell_ids = cell_ids
        
        # load input datas
        self.input_datas = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["INPUT"]):
            key = datas["NAME"]
            if datas["DATA"] == "csv":
                self.input_datas[key] = self.load_data_csv(self.data_root, datas)
            if datas["DATA"] == "numpy":
                self.input_datas[key] = self.load_data_numpy(self.data_root, datas)
            # add smote
            if cfg["DATASET"].get("TRAINSMOTE", None) is not None and mode=='train':
                for smote_file in cfg["DATASET"]["TRAINSMOTE"]:
                    smote_data = self.load_data_csv(self.data_root, datas, smote_file)
                    self.input_datas[key] = np.concatenate((self.input_datas[key], smote_data))

        # print("[ load input datas ]")
        # for k, v in self.input_datas.items():
        #     print("... {:<20} | {}".format(k, v.shape))
        # exit()
        
        
        # load output datas
        self.output_datas = {}
        if mode != 'inference':
            for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
                key = datas["NAME"]
                if datas["DATA"] == "csv":
                    self.output_datas[key] = self.load_data_csv(self.data_root, datas)
                # add smote
                if cfg["DATASET"].get("TRAINSMOTE", None) is not None and mode=='train':
                    for smote_file in cfg["DATASET"]["TRAINSMOTE"]:
                        smote_data = self.load_data_csv(self.data_root, datas, smote_file)
                        self.output_datas[key] = np.concatenate((self.output_datas[key], smote_data))

        # print("[ load output datas ]")
        # for k, v in self.output_datas.items():
        #     print("... {:<20} | {}".format(k, v.shape))
        # print("---" * 20)
        # exit()

        # load transfrom
        self.transform_input  = TransformCell(self.cfg["DATASET"]["INPUT"])
        self.transform_output = TransformCell(self.cfg["DATASET"]["OUTPUT"])

    def load_data_numpy(self, root, cfg):
        data_dir = os.path.join(root, cfg['DATA'], cfg['TYPE']) 
        datas = []
        for cell_id in self.cell_ids:
            np_data = os.path.join(data_dir, cell_id+".npy")
            np_data = np.load(np_data)
            datas.append(np_data)
        datas = np.stack(datas)
        return datas

    def load_data_csv(self, root, cfg, csv_file=None):
        # read csv
        if csv_file is None:
            csv_file = os.path.join(root, cfg['DATA'], cfg['TYPE']+".csv")
        csv_data = pd.read_csv(csv_file)
        # get cell id and data idx
        cell_id2data_idx = {}
        for data_idx, cell_id in enumerate(csv_data['cell_id']):
            cell_id2data_idx[cell_id] = data_idx
        # get data in cfg['ITEM']
        datas = [] # shape : (cell_ids, items)
        for cell_id in self.cell_ids:
            if cell_id not in cell_id2data_idx: continue
            data_idx = cell_id2data_idx[cell_id]
            data = [csv_data[item][data_idx] for item in cfg['ITEM']]
            datas.append(data)
        datas = np.array(datas)
        return datas

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        # load input data
        data_input = {}
        for k, v in self.input_datas.items():
            data_input[k] = v[idx]
        # load output data
        data_output = {}
        for k, v in self.output_datas.items():
            data_output[k] = v[idx]
        # get transform
        data_input  = self.transform_input.norm(data_input)
        data_output = self.transform_output.norm(data_output)
        return data_input, data_output




class BatteryCellDataSMOTE(Dataset):
    def __init__(self, cfg, mode="train", seed=777):
        assert mode in ["train", "test", "inference"]
        self.mode = mode

        # set seed for reproduciblity
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.cfg = copy.deepcopy(cfg)
        self.data_root = self.cfg["DATASET"]["ROOT"]
        
        dir_smote = 'csv_smote'
        dir_smote = 'csv_smote_small'
        dir_smote = 'csv_smote_dp'
        dir_smote = 'csv_smote_sc'
        dir_smote = 'csv_smote_0.1_0.2C'
        # load input datas
        self.input_datas = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["INPUT"]):
            key = datas["NAME"]
            if datas["DATA"] == "csv":
                datas["DATA"] = dir_smote
                self.input_datas[key] = self.load_data_csv_smote(self.data_root, datas)
        print("[ load input datas ]")
        for k, v in self.input_datas.items():
            print("... {:<20} | {}".format(k, v.shape))

        # load output datas
        self.output_datas = {}
        if mode != 'inference':
            for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
                key = datas["NAME"]
                if datas["DATA"] == "csv":
                    datas["DATA"] = dir_smote
                    self.output_datas[key] = self.load_data_csv_smote(self.data_root, datas)
        print("[ load output datas ]")
        for k, v in self.output_datas.items():
            print("... {:<20} | {}".format(k, v.shape))
        print("---" * 20)

        # set len dataset
        self.num_data = len(self.output_datas[key])
        # load transfrom
        self.transform_input  = TransformCell(self.cfg["DATASET"]["INPUT"])
        self.transform_output = TransformCell(self.cfg["DATASET"]["OUTPUT"])

    def load_data_csv_smote(self, root, cfg):
        # read csv
        csv_root = os.path.join(root, cfg['DATA'])
        csv_list = sorted(os.listdir(csv_root))
        datas = [] # shape : (cells, items)
        for csv_name in tqdm(csv_list):
            csv_file = os.path.join(csv_root, csv_name)
            csv_data = pd.read_csv(csv_file)
            data = [csv_data[item].to_numpy() for item in cfg['ITEM']]
            data = np.array(data).transpose()
            datas.append(data)
            # print("... [{:<40s}] {} {}".format(csv_name, len(csv_data), data.shape))
        datas = np.concatenate(datas, axis=0)
        return datas

    def load_data_csv(self, root, cfg):
        # read csv
        csv_file = os.path.join(root, cfg['DATA'], cfg['TYPE']+".csv")
        csv_data = pd.read_csv(csv_file)
        # get cell id and data idx
        cell_id2data_idx = {}
        for data_idx, cell_id in enumerate(csv_data['cell_id']):
            cell_id2data_idx[cell_id] = data_idx
        # get data in cfg['ITEM']
        datas = [] # shape : (cell_ids, items)
        for cell_id in self.cell_ids:
            data_idx = cell_id2data_idx[cell_id]
            data = [csv_data[item][data_idx] for item in cfg['ITEM']]
            datas.append(data)
        datas = np.array(datas)
        return datas

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # load input data
        data_input = {}
        for k, v in self.input_datas.items():
            data_input[k] = v[idx]
        # load output data
        data_output = {}
        for k, v in self.output_datas.items():
            data_output[k] = v[idx]
        # get transform
        data_input  = self.transform_input.norm(data_input)
        data_output = self.transform_output.norm(data_output)
        return data_input, data_output
