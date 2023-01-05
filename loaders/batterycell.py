from calendar import c
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from .transform import *

import math

class MetaDataGen(Dataset):
    def __init__(self, ind_vals):

        self.div_factors = {
            'cathod_전극 두께 (㎛)': 1000,
            'anode_전극 두께 (㎛)': 100,
            'cathod_AM thickness (㎛)': 1000,
            'anode_AM thickness (㎛)': 100,
            'cathod_Weight (mg)': 100,
            'anode_Weight (mg)': 10,
            'cathod_Electrode weight (w/o foil,g)': 0.1,
            'anode_Electrode weight (w/o foil,g)': 0.1,
            'cathod_Loading mass of AM (mg)': 100,
            'anode_Loading mass of AM (mg)': 10,
            'cathod_Loading (mg/cm2)': 10,
            'anode_Loading (mg/cm2)': 10,
            'cathod_Porosity': 1,
            'anode_Porosity': 1,
            'cathod_Loading density (mAh/cm2)': 10,
            'anode_Loading density (mAh/cm2)': 10,
            'cathod_Theoretical capacity (mAh)': 10000,
            'anode_Theoretical capacity (mAh)': 10000
            }

        for k in ind_vals.keys():
            ind_vals[k] *= self.div_factors[k]

        dep_vals = {}
        # anode meta
        if 'anode_전극 두께 (㎛)' in ind_vals and 'anode_Weight (mg)' in ind_vals:
            dep_vals['anode_전극 두께 (㎛)'] = ind_vals['anode_전극 두께 (㎛)']
            dep_vals['anode_AM thickness (㎛)'] = dep_vals['anode_전극 두께 (㎛)']-30
            dep_vals['anode_Weight (mg)'] = ind_vals['anode_Weight (mg)']
            dep_vals['anode_Electrode weight (w/o foil,g)'] = (dep_vals['anode_Weight (mg)']-6.4)/1000
            dep_vals['anode_Loading mass of AM (mg)'] = round((dep_vals['anode_Weight (mg)']-6.4)*0.93,2)
            dep_vals['anode_Loading (mg/cm2)'] = dep_vals['anode_Electrode weight (w/o foil,g)']*1000/(0.25*math.pi)

            # dep_vals['anode_Porosity'] = (dep_vals['anode_AM thickness (㎛)']/10000)*math.pi*(0.25)
            # dep_vals['anode_Porosity'] -= dep_vals['anode_Electrode weight (w/o foil,g)']*0.93/4.7
            # dep_vals['anode_Porosity'] -= dep_vals['anode_Electrode weight (w/o foil,g)']*0.04/1.9
            # dep_vals['anode_Porosity'] -= dep_vals['anode_Electrode weight (w/o foil,g)']*0.03/1.78
            # dep_vals['anode_Porosity'] /= (dep_vals['anode_AM thickness (㎛)']/10000)*math.pi*(0.25)
            dep_vals['anode_Porosity']  = dep_vals['anode_Electrode weight (w/o foil,g)']
            dep_vals['anode_Porosity'] /= dep_vals['anode_AM thickness (㎛)']
            dep_vals['anode_Porosity'] *= 10000 / math.pi / 0.25
            dep_vals['anode_Porosity'] *= (0.93/4.7) + (0.04/1.9) + (0.03/1.78)
            dep_vals['anode_Porosity'] = 1 - dep_vals['anode_Porosity']

            dep_vals['anode_Loading density (mAh/cm2)'] = round((dep_vals['anode_Loading mass of AM (mg)']*210),-1)/(0.25*math.pi*1000)
            dep_vals['anode_Theoretical capacity (mAh)'] = round((dep_vals['anode_Loading mass of AM (mg)']*210),-1)

        elif 'anode_Weight (mg)' in ind_vals and 'anode_Porosity' in ind_vals:
            dep_vals['anode_Weight (mg)'] = ind_vals['anode_Weight (mg)']
            dep_vals['anode_Electrode weight (w/o foil,g)'] = (dep_vals['anode_Weight (mg)']-6.4)/1000
            dep_vals['anode_Loading mass of AM (mg)'] = round((dep_vals['anode_Weight (mg)']-6.4)*0.93,2)
            dep_vals['anode_Loading (mg/cm2)'] = dep_vals['anode_Electrode weight (w/o foil,g)']*1000/(0.25*math.pi)
            dep_vals['anode_Loading density (mAh/cm2)'] = round((dep_vals['anode_Loading mass of AM (mg)']*210),-1)/(0.25*math.pi*1000)
            dep_vals['anode_Theoretical capacity (mAh)'] = round((dep_vals['anode_Loading mass of AM (mg)']*210),-1)

            dep_vals['anode_Porosity'] = ind_vals['anode_Porosity']
            dep_vals['anode_AM thickness (㎛)']  = dep_vals['anode_Electrode weight (w/o foil,g)']
            dep_vals['anode_AM thickness (㎛)'] /= 1 - dep_vals['anode_Porosity']
            dep_vals['anode_AM thickness (㎛)'] *= 10000 / math.pi / 0.25
            dep_vals['anode_AM thickness (㎛)'] *= (0.93/4.7) + (0.04/1.9) + (0.03/1.78)

            dep_vals['anode_전극 두께 (㎛)'] = dep_vals['anode_AM thickness (㎛)'] + 30

        # cathod meta
        if 'cathod_전극 두께 (㎛)' in ind_vals and 'cathod_Weight (mg)' in ind_vals:
            dep_vals['cathod_전극 두께 (㎛)'] = ind_vals['cathod_전극 두께 (㎛)']
            dep_vals['cathod_AM thickness (㎛)'] = dep_vals['cathod_전극 두께 (㎛)']-30
            dep_vals['cathod_Weight (mg)'] = ind_vals['cathod_Weight (mg)']
            dep_vals['cathod_Electrode weight (w/o foil,g)'] = (dep_vals['cathod_Weight (mg)']-45)/1000
            dep_vals['cathod_Loading mass of AM (mg)'] = round((dep_vals['cathod_Weight (mg)']-45)*0.96,2)
            dep_vals['cathod_Loading (mg/cm2)'] = dep_vals['cathod_Electrode weight (w/o foil,g)']*1000/(0.49*math.pi)

            dep_vals['cathod_Porosity'] = (dep_vals['cathod_AM thickness (㎛)']/10000)*math.pi*(0.49)
            dep_vals['cathod_Porosity'] -= dep_vals['cathod_Electrode weight (w/o foil,g)']*(0.96/2.26) 
            dep_vals['cathod_Porosity'] -= dep_vals['cathod_Electrode weight (w/o foil,g)']*0.01/1.95
            dep_vals['cathod_Porosity'] -= dep_vals['cathod_Electrode weight (w/o foil,g)']*0.015/1.254
            dep_vals['cathod_Porosity'] /= (dep_vals['cathod_AM thickness (㎛)']/10000)*math.pi*(0.49)
            # dep_vals['cathod_Porosity']  = dep_vals['cathod_Electrode weight (w/o foil,g)']
            # dep_vals['cathod_Porosity'] /= dep_vals['cathod_AM thickness (㎛)']
            # dep_vals['cathod_Porosity'] *= 10000 / math.pi / 0.49
            # dep_vals['cathod_Porosity'] *= (0.96/2.26) + (0.01/1.95) + (0.015/1.254)
            # dep_vals['cathod_Porosity'] = 1 - dep_vals['cathod_Porosity']

            print(dep_vals['cathod_전극 두께 (㎛)'])
            print(dep_vals['cathod_AM thickness (㎛)'])
            print(dep_vals['cathod_Electrode weight (w/o foil,g)'])
            print(dep_vals['cathod_Porosity'])

            dep_vals['cathod_Loading density (mAh/cm2)'] = round((dep_vals['cathod_Loading mass of AM (mg)']*360),-1)/(0.49*math.pi*1000)
            dep_vals['cathod_Theoretical capacity (mAh)'] = round((dep_vals['cathod_Loading mass of AM (mg)']*360),-1)

        elif 'cathod_Weight (mg)' in ind_vals and 'cathod_Porosity' in ind_vals:
            dep_vals['cathod_Weight (mg)'] = ind_vals['cathod_Weight (mg)']
            dep_vals['cathod_Electrode weight (w/o foil,g)'] = (dep_vals['cathod_Weight (mg)']-45)/1000
            dep_vals['cathod_Loading mass of AM (mg)'] = round((dep_vals['cathod_Weight (mg)']-45)*0.96,2)
            dep_vals['cathod_Loading (mg/cm2)'] = dep_vals['cathod_Electrode weight (w/o foil,g)']*1000/(0.49*math.pi)
            dep_vals['cathod_Loading density (mAh/cm2)'] = round((dep_vals['cathod_Loading mass of AM (mg)']*360),-1)/(0.49*math.pi*1000)
            dep_vals['cathod_Theoretical capacity (mAh)'] = round((dep_vals['cathod_Loading mass of AM (mg)']*360),-1)

            dep_vals['cathod_Porosity'] = ind_vals['cathod_Porosity']
            dep_vals['cathod_AM thickness (㎛)']  = dep_vals['cathod_Electrode weight (w/o foil,g)']
            dep_vals['cathod_AM thickness (㎛)'] /= 1 - dep_vals['cathod_Porosity']
            dep_vals['cathod_AM thickness (㎛)'] *= 10000 / math.pi / 0.49
            dep_vals['cathod_AM thickness (㎛)'] *= (0.96/2.26) + (0.01/1.95) + (0.015/1.254)

            dep_vals['cathod_전극 두께 (㎛)'] = dep_vals['cathod_AM thickness (㎛)'] + 30


        for k in dep_vals.keys():
            dep_vals[k] /= self.div_factors[k]
            
        self.ind_vals = ind_vals
        self.dep_vals = dep_vals


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


class BatteryCellData(Dataset):
    def __init__(self, cfg, k_fold=None, mode="train", seed_np=777):
        self.cfg = cfg
        assert mode in ["train", "test", "inverse_test"]
        self.data_root = cfg["DATASET"]["ROOT"]

        # check cell ids existence for INPUT and OUTPUT
        self.get_cell_ids()
        
        print("DATASET >>> K-FOLD:", k_fold)
        if k_fold is not None:
            # get cell sample id (X.Y) with K-FOLD and MODE (train, test)
            cell_sample = "{}.{}".format(k_fold // 3 + 1, k_fold % 3 + 1)
            print("[KFOLD {}] - Sample {}".format(k_fold, cell_sample))
            # get cell ids (X.Y-Z)
            if mode.endswith("test"):
                self.cell_ids = [cell for cell in self.cell_ids
                                if cell.split("-")[0] == cell_sample]
            else:
                self.cell_ids = [cell for cell in self.cell_ids 
                                if cell.split("-")[0] != cell_sample]
            self.cell_ids = sorted(self.cell_ids)
        else:
            # devide cell ids equally (8:1=Train:Test)
            if "SEED" in cfg["DATASET"]: seed_np = cfg["DATASET"]["SEED"]
            np.random.seed(seed_np)
            np.random.shuffle(self.cell_ids)
            num_data = int(len(self.cell_ids)/9)
            if mode.endswith("test"): self.cell_ids = self.cell_ids[:num_data]
            else: self.cell_ids = self.cell_ids[num_data:]
            self.cell_ids = sorted(self.cell_ids)
        print("[DATA] <{}> mode - {} cells".format(mode, len(self.cell_ids)))

        # load input data
        self.input_datas = {}
        self.transforms = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["INPUT"]):
            if datas["DATA"] == "META":
                meta_file = os.path.join(
                    self.cfg["DATASET"]["ROOT"], "pickle",
                    'cell_meta_{}.pickle'.format(datas["TYPE"]))
                data_items = datas["ITEM"]
                key = "{}_{}".format(datas["DATA"], data_idx)
                self.input_datas[key] = self.load_data_meta(meta_file, data_items)
            else: # CYCLE, TIME
                data_len = datas["LEN"] if "LEN" in datas else None
                data_root = os.path.join(self.cfg["DATASET"]["ROOT"], "numpy", 
                                datas["DATA"].lower(), datas["TYPE"], datas["ITEM"])
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
                self.input_datas[key] = self.load_data_numpy(data_root, data_len)
                # load transform for time seq data
                if mode == "train" and "TRNSFORM" in datas:
                    self.transforms[key] = self.load_transform(datas["TRNSFORM"])

        # load output data
        self.output_datas = {}
        for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
            if datas["DATA"] == "META":
                meta_file = os.path.join(
                    self.cfg["DATASET"]["ROOT"], "pickle",
                    'cell_meta_{}.pickle'.format(datas["TYPE"]))
                data_items = datas["ITEM"]
                key = "{}_{}".format(datas["DATA"], data_idx)
                self.output_datas[key] = self.load_data_meta(meta_file, data_items)
            else: # CYCLE, TIME
                data_len = datas["LEN"] if "LEN" in datas else None
                data_root = os.path.join(self.cfg["DATASET"]["ROOT"], "numpy", 
                                datas["DATA"].lower(), datas["TYPE"], datas["ITEM"])
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
                self.output_datas[key] = self.load_data_numpy(data_root, data_len)

        print("... [INPUT DATA]")
        for k, v in self.input_datas.items():
            print("......", k, v.shape)

        print("... [OUTPUT DATA]")
        for k, v in self.output_datas.items():
            print("......", k, v.shape)

        self.num_gt = len(self.cell_ids)

        # generate random data for inverse test
        if mode == "inverse_test":
            self.generate_random_data()
            # pass
       
    def __len__(self):
        return len(self.cell_ids)

    def generate_random_data(self):

        print("---" * 20)
        print("+++ generate random dataset")
        num_sampling = 2
        num_generate = 2

        for datas in self.cfg["DATASET"]["INPUT"]:
            if datas["DATA"] == "META": break
        items = datas["ITEM"]


        sampling_range = 0.00
        target_keys = ['anode_Weight (mg)', 'anode_Porosity',
                       'cathod_Weight (mg)', 'cathod_Porosity']

        target_keys = ['anode_전극 두께 (㎛)', 'anode_Weight (mg)',
                       'cathod_전극 두께 (㎛)', 'cathod_Weight (mg)']

        target_ids  = {k:items.index(k) for k in target_keys}
        target_datas = {k:[] for k in target_keys}

        self.target_keys = target_keys
        self.target_ids = target_ids

        for input_key, input_datas in self.input_datas.items():
            # find [전극 두께] and [Weight]
            for input_data in input_datas:
                for k in target_keys:
                    idx = target_ids[k]
                    data = input_data[idx]
                    target_datas[k].append(data)
    
            # sampling in range of [전극 두께] and [Weight]
            sampling_datas = {k:[] for k in target_keys}
            for k in target_keys:
                target_datas[k] = np.array(target_datas[k])
                min_val = target_datas[k].min()
                max_val = target_datas[k].max()
                min_val *= 1-sampling_range
                max_val *= 1+sampling_range
                gap_val = (max_val-min_val) / num_sampling
                sampling_datas[k] = np.arange(min_val, max_val, gap_val)
                print("[{}] min: {:.3f} | max: {:.3f}".format(
                      k, min_val, max_val))
                # sampling_datas[k] = np.array(target_datas[k]).copy()
            # generate other datas (dependent values)
            generate_datas = {k:[] for k in items}
            for data_idx in range(num_generate):
                ind_vals = {}
                print("---" * 20)
                for k in target_keys:
                    data = np.random.choice(sampling_datas[k], 1)[0]
                    ind_vals[k] = data
                    print(k)

                # TODO: delete after debuggign
                # ind_vals = {}
                # ind_vals['anode_전극 두께 (㎛)'] = 0.618
                # ind_vals['anode_Weight (mg)'] = 1.68
                # ind_vals['anode_Porosity'] = 0.197401
                # ind_vals['cathod_전극 두께 (㎛)'] = 0.073
                # ind_vals['cathod_Weight (mg)'] = 0.554
                # ind_vals['cathod_Porosity'] = 0.28068

                print(ind_vals)
                meta_gen = MetaDataGen(ind_vals).dep_vals
                print("---" * 20)
                for gen_k, gen_val in meta_gen.items():
                    generate_datas[gen_k].append(gen_val)
                    print(gen_k, gen_val)

            # parsing as dataset
            gen_input_datas = []
            for data_idx in range(num_generate):
                gen_input_data = []
                for k in items:
                    gen_input_data.append(generate_datas[k][data_idx])
                gen_input_datas.append(gen_input_data)
            gen_input_datas = np.array(gen_input_datas)
            self.input_datas[input_key] = np.vstack(
                                            [self.input_datas[input_key], 
                                            gen_input_datas])
            gen_cell_ids = ["gen_{}".format(idx) for idx in range(num_generate)]
            self.cell_ids += gen_cell_ids

    def load_transform(self, cfg):
        if cfg.get("ITEM", "partial") == "full":
            transform = Compose(
                            transforms=[
                                RandomAmplitudeScale(),
                                RandomTimeShift(),
                                RandomDCShift(),
                                RandomZeroMasking(),
                                RandomAdditiveGaussianNoise(),
                                ],
                            mode=cfg["mode"]
                            )
        else:
            transform = Compose(
                            transforms=[
                                RandomAdditiveGaussianNoise(),
                                ],
                            mode=cfg["mode"]
                            )
        return transform

    def get_cell_ids(self):
        cell_ids_all = None
        for datas in self.cfg["DATASET"]["INPUT"]+self.cfg["DATASET"]["OUTPUT"]:
            if datas["DATA"] == "META":
                meta_file = os.path.join(self.cfg["DATASET"]["ROOT"], "pickle",
                                'cell_meta_{}.pickle'.format(datas["TYPE"]))
                                # 'cell_meta_{}_new.pickle'.format(datas["TYPE"]))
                with open(meta_file, 'rb') as f:
                    cell_ids = pickle.load(f).keys()
            else: # CYCLE, TIME
                data_root = os.path.join(self.cfg["DATASET"]["ROOT"], "numpy", 
                                datas["DATA"].lower(), datas["TYPE"], datas["ITEM"])
                cell_ids = [cell_id.replace(".npy", "") for cell_id in os.listdir(data_root)]
            if cell_ids_all is None: cell_ids_all = cell_ids
            else: cell_ids_all = [cell_id for cell_id in cell_ids if cell_id in cell_ids_all]
        self.cell_ids = cell_ids_all

    def load_cell_id(self, mode, label_or_id="label"):
        assert label_or_id in ["label", "id", "both"]
        id2label = {}
        cell_ids = []
        cell_labels = []
        for cell_id in self.cell_ids:
            if mode == "X": cell_id = cell_id[0]
            elif mode == "Y": cell_id = cell_id[2]
            elif mode == "X.Y": cell_id = cell_id[:3]
            else: raise ValueError("Wrong type of CELL ID data from config file[{}]".format(mode))
            if cell_id not in id2label: id2label[cell_id] = len(id2label)
            cell_label = id2label[cell_id]
            cell_labels.append(cell_label)
        cell_ids = np.array(cell_ids)
        cell_labels = np.array(cell_labels).astype(np.long)
        if label_or_id == "label": return cell_labels
        if label_or_id == "id": return cell_ids
        else: return cell_ids, cell_labels

    def load_data_numpy(self, root, data_len=None):
        # load numpy datas
        datas = []
        for cell_id in self.cell_ids:
            np_data = os.path.join(root, cell_id+".npy")
            np_data = np.load(np_data)
            # TODO: CHANGE HERE !
            if data_len is not None:
                if len(np_data) >= data_len: np_data = np_data[:data_len]
                else: print("Error SHORTER THAN LEN !", root, cell_id, np_data.shape, data_len)
            datas.append(np_data)
        datas = np.stack(datas)
        return datas

    def load_data_meta(self, meta_file, data_items):
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        datas = []
        for cell_id in self.cell_ids:
            data = [meta[cell_id][data_item] for data_item in data_items]
            datas.append(data)
        datas = np.array(datas)
        return datas

    def __getitem__(self, idx):
        # load input data
        data_input = {}
        for k, v in self.input_datas.items():
            if k in self.transforms:
                data_input[k] = self.transforms[k](v[idx])
            else:
                data_input[k] = v[idx]
        # load output data
        data_output = {}
        for k, v in self.output_datas.items():
            if idx >= len(v):
                idx = idx % len(v)
            data_output[k] = v[idx]
        return data_input, data_output