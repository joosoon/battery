import os
from tokenize import Whitespace
from typing import Counter
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def save_meta_as_pickle(file_name, sheet_name, save_name, 
                        meta_items, del_cells=None):
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    # read and extract production data
    data_meta = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl', header=18)
    # fill sample id (1.1 ~ 3.3)
    last = None
    for i, v in enumerate(tqdm(data_meta["Sample"])):
        if v == v: # not NaN
            last = v
        else:
            data_meta["Sample"][i] = last

    cell2data = {}
    for i in tqdm(range(len(data_meta))):
        # skip empty row
        if data_meta["Number"][i] != data_meta["Number"][i]:
            continue
        cell_id = "{}-{}".format(data_meta["Sample"][i], int(data_meta["Number"][i]))
        # skip if in del list
        if del_cells is not None and cell_id in del_cells:
            continue
        cell2data[cell_id] = {}
        for item in meta_items:
            value = data_meta[item][i]
            if item[-2:] != ".1":
                key = "anode_" + item
            else:
                key = "cathod_" + item[:-2]
            cell2data[cell_id][key] = value
    # save
    with open(save_name, 'wb') as f:
        pickle.dump(cell2data, f, pickle.HIGHEST_PROTOCOL)
    # load and test
    with open(save_name, 'rb') as f:
        data = pickle.load(f)
        print(type(data), len(data))

def get_range_meta(file_name, meta_items):
    with open(file_name, 'rb') as f:
        meta_data = pickle.load(f)
    # get mean range of each items
    meta2data = {}
    for meta_item in meta_items:
        c_item = "cathod_"+meta_item
        a_item = "anode_"+meta_item
        if c_item not in meta2data:
            meta2data[c_item] = []
        if a_item not in meta2data:
            meta2data[a_item] = []
    # load meta value
    for cell, data in meta_data.items():
        for meta_item in meta_items:
            c_item = "cathod_"+meta_item
            a_item = "anode_"+meta_item
            c_data = data[c_item]
            a_data = data[a_item]
            if c_data == c_data: # add if not None
                meta2data[c_item].append(c_data)
            if a_data == a_data: # add if not None
                meta2data[a_item].append(a_data)
    # convert LIST to NUMPY
    print("[SHAPE] | [MEAN, STD] | [MIN, MAX] | ITEM")
    for meta_item in meta2data.keys():
        meta2data[meta_item] = np.array(meta2data[meta_item])
        data = meta2data[meta_item]
        print(data.shape, "| {:.3f}, {:.3f} | {:.2f}, {:.2f} | {}".format(
                np.abs(data).mean(), np.abs(data).std(), 
                np.abs(data).min(), np.abs(data).max(), meta_item))

def re_range_meta(file_name, save_name, meta_means, 
                  cell_list=None, del_list=None):
    with open(file_name, 'rb') as f:
        meta_data = pickle.load(f)
    meta_data_new = {}
    for cell, data in meta_data.items():
        if del_list is not None and cell in del_list: continue
        if cell_list is not None and cell not in cell_list: continue
        meta_data_new[cell] = {}
        for data_item, data_value in data.items():
            mean_value = meta_means[data_item]
            meta_data_new[cell][data_item] = data_value / mean_value
    # save
    with open(save_name, 'wb') as f:
        pickle.dump(meta_data_new, f, pickle.HIGHEST_PROTOCOL)
    # load and test
    with open(save_name, 'rb') as f:
        data = pickle.load(f)
        print(type(data), len(data))

def extract_cell_data(data_root, save_root, del_cells=None):
    file_list = sorted(os.listdir(data_root))
    for file_idx, file_name in enumerate(tqdm(file_list)):
    # for file_idx, file_name in enumerate(file_list):
    #     print()
    #     print("[{}/{}] {}".format(file_idx, len(file_list), file_name))
        
        # load excel
        if file_name.startswith("."): continue
        # data_excel = pd.ExcelFile(os.path.join(data_root, file_name))
        data_excel = pd.ExcelFile(os.path.join(data_root, file_name), engine='openpyxl')
        sheet_names = data_excel.sheet_names

        # matching sheet names    
        sheet_meta = [name for name in sheet_names if name[:2] == "정보"]
        sheet_data = [name for name in sheet_names if name[:2] != "정보"]
        meta2data = {}
        for meta_name in sheet_meta:
            sheet_id = meta_name.split("_")[1]
            for data_name in sheet_data:
                if data_name.split("_")[1] == sheet_id:
                    meta2data[meta_name] = data_name
        assert len(sheet_meta) == len(sheet_data)
        assert len(sheet_meta) == len(meta2data)

        # for sheet_name in sheet_meta:
        for sheet_name in tqdm(sheet_meta):
            # get cell identifier
            sheet_id = int(sheet_name.split("_")[1])
            data_sheet = data_excel.parse(sheet_name)
            heads = list(data_sheet.columns)
            cell_id = data_sheet[heads[0]][6][11:].split("\\")
            cell_id = cell_id[-1].split(" ")[1] # X.Y-N
    
            # skip if in del list
            if del_cells is not None and cell_id in del_cells:
                continue

            data_sheet = data_excel.parse(meta2data[sheet_name])
            datas = list(data_sheet.columns)

            assert datas[0] == "인덱스"
            assert datas[1] in ["사이클 횟수", "시험 시간(s)"]
            # for data_type in datas[2:]:
            for data_type in datas[1:]:
                # convert data as numpy
                data_np = np.array(data_sheet[data_type].tolist())
                # print(data_type, data_np.shape)
                # rename data type
                data_type = data_type.replace("/g", "")
                data_type = data_type.replace("_s", "")
                data_type = data_type.replace("/", "-")
                # save raw datas ad numpy 
                save_dir = os.path.join(save_root, data_type)
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, "{}".format(cell_id)), data_np)

def count_data(data_root):
    data_dirs = os.listdir(data_root)
    for data_dir in data_dirs:
        f_list = os.listdir(os.path.join(data_root, data_dir))
        print(len(f_list), data_dir)

def normalize_capa(cell2meta, data_items, data_root, save_root):
    for data_item in tqdm(data_items):
        # save directory
        save_dir = os.path.join(save_root, data_item)
        os.makedirs(save_dir, exist_ok=True)
        # load numpy files
        data_dir = os.path.join(data_root, data_item)
        file_list = sorted(os.listdir(data_dir))
        for file_name in file_list:
            cell_id = file_name[:-4]
            # load numpy data
            data_np = os.path.join(data_root, data_item, file_name)
            data_np = np.load(data_np)
            # dividing with loading mass (from META DATA)
            loading_mass =  cell2meta[cell_id]["anode_Loading mass of AM (mg)"]
            data_np_norm = data_np / loading_mass
            # print(cell_id, loading_mass, "{:.6f} {:.6f}".format(data_np.mean(), data_np_norm.mean()), data_np_norm.shape)
            # save numpy data 
            np.save(os.path.join(save_dir, cell_id+".npy"), data_np_norm)

def avg_norm_cycle_data(data_root, save_root, avg_list):
    os.makedirs(save_root, exist_ok=True)
    file_list = sorted(os.listdir(data_root))
    for file_name in tqdm(file_list):
        # load data
        data_np = os.path.join(data_root, file_name)
        data_np = np.load(data_np)
        # load idx to average
        cell = file_name[:-4]
        if cell not in avg_list: continue
        ids = avg_list[cell]
        # average datas
        datas = []
        for i in range(8):
            data = data_np[ids[i]].mean()
            datas.append(data)
        datas = np.array(datas)
        # dividing with max-value (first value)
        datas /= datas[0]
        # save numpy
        save_name = os.path.join(save_root, file_name)
        np.save(save_name, datas)

def get_idx_cycles_with_current(arr):
    idxs = np.where(arr == 0)[0]
    pts = []
    for i in range(len(idxs)-1):
        diff = idxs[i+1] - idxs[i]
        if diff > 1: 
            pts.append([idxs[i]+1, idxs[i+1]])
    return np.array(pts)


if __name__ == '__main__':
    ###############################
    # 0. excluding defective data #
    ###############################
    DEL_CELLS = [
        '1.1-20', '1.2-28', '1.3-13', '2.1-12', 
        '2.1-2', '2.1-26', '2.1-3', '2.1-6', 
        '2.2-12', '2.3-2', '2.3-31', '3.1-4', 
        '3.1-6', '3.2-27']

    ###########
    # 1. META # 
    ############################################
    # 1-1. extract meta data as .pickle format #
    ############################################
    # # save as pickle
    # file_name = "./data/excel/schedule.xlsx"
    # sheet_name = "AI"
    # save_name = './data/pickle/cell_meta_raw.pickle'
    # META_ITEMS = [
    #     '전극 두께 (㎛)', 'AM thickness (㎛)', 'Weight (mg)', 
    #     'Electrode weight (w/o foil,g)', 'Loading mass of AM (mg)', 
    #     'Loading (mg/cm2)', 'Porosity', 'Loading density (mAh/cm2)', 
    #     'Theoretical capacity (mAh)', 
    #     '전극 두께 (㎛).1', 'AM thickness (㎛).1', 'Weight (mg).1', 
    #     'Electrode weight (w/o foil,g).1', 'Loading mass of AM (mg).1', 
    #     'Loading (mg/cm2).1', 'Porosity.1', 'Loading density (mAh/cm2).1', 
    #     'Theoretical capacity (mAh).1' ]
    # save_meta_as_pickle(file_name, sheet_name, save_name, 
    #                     META_ITEMS, del_cells=DEL_CELLS)

    # # re-ranging META DATA / range: [0, 1]
    # file_name = './data/pickle/cell_meta_raw.pickle'
    # META_ITEMS = ['전극 두께 (㎛)', 'AM thickness (㎛)', 'Weight (mg)', 
    #               'Electrode weight (w/o foil,g)', 'Loading mass of AM (mg)', 
    #               'Loading (mg/cm2)', 'Porosity', 'Loading density (mAh/cm2)', 
    #               'Theoretical capacity (mAh)']
    # get_range_meta(file_name, META_ITEMS)
    # save_name = './data/pickle/cell_meta_norm.pickle'
    # META_MEANS = {
    #     'cathod_전극 두께 (㎛)': 1000,
    #     'anode_전극 두께 (㎛)': 100,
    #     'cathod_AM thickness (㎛)': 1000,
    #     'anode_AM thickness (㎛)': 100,
    #     'cathod_Weight (mg)': 100,
    #     'anode_Weight (mg)': 10,
    #     'cathod_Electrode weight (w/o foil,g)': 0.1,
    #     'anode_Electrode weight (w/o foil,g)': 0.1,
    #     'cathod_Loading mass of AM (mg)': 100,
    #     'anode_Loading mass of AM (mg)': 10,
    #     'cathod_Loading (mg/cm2)': 10,
    #     'anode_Loading (mg/cm2)': 10,
    #     'cathod_Porosity': 1,
    #     'anode_Porosity': 1,
    #     'cathod_Loading density (mAh/cm2)': 10,
    #     'anode_Loading density (mAh/cm2)': 10,
    #     'cathod_Theoretical capacity (mAh)': 10000,
    #     'anode_Theoretical capacity (mAh)': 10000}
    # del_list = ["1.1-20", "1.2-28", "2.1-12", "2.1-26", "2.3-31", "3.2-27"]
    # re_range_meta(file_name, save_name, META_MEANS)
    # exit()


    ############
    # 2. CYCLE #
    ################################################
    # 2-1. extract raw data sequence for each cell #
    ################################################
    # data_root = "./data/excel/cycle"
    # save_root = "./data/numpy/cycle/raw"
    # extract_cell_data(data_root, save_root, del_cells=DEL_CELLS)
    # count_data(save_root)
    # exit()

    ###########################################################
    # 2-2. dividing raw CAPACITY data with LOADING MASS OF AM #
    ###########################################################
    # meta_file = "./data/pickle/cell_meta_raw.pickle"
    # with open(meta_file, 'rb') as f:
    #     cell2meta = pickle.load(f)

    # data_root = "./data/numpy/cycle/raw"
    # save_root = "./data/numpy/cycle/norm"
    # DATA_ITMES_CYCLE = ['충전 용량(Ah)', '방전 용량(Ah)']
    # normalize_capa(cell2meta, DATA_ITMES_CYCLE, data_root, save_root)
    # count_data(save_root)

    #####################################################
    # 2-3. averaging each C-RATE and dividing with 0.1C #
    #####################################################
    # data_root = "./data/numpy/cycle/norm/충전 용량(Ah)"
    # save_root = "./data/numpy/cycle/avg/충전 용량(Ah)"
    # from preprocess_CYCLE_AVG_rev220306 import ids_1 as avg_list
    # avg_norm_cycle_data(data_root, save_root, avg_list)

    # data_root = "./data/numpy/cycle/norm/방전 용량(Ah)"
    # save_root = "./data/numpy/cycle/avg/방전 용량(Ah)"
    # from preprocess_CYCLE_AVG_rev220306 import ids_1 as avg_list
    # avg_norm_cycle_data(data_root, save_root, avg_list)

    # save_root = "./data/numpy/cycle/avg"
    # count_data(save_root)

    #######################
    # 2-4. analysis power #
    #######################



    ###########
    # 3. TIME #
    ################################################
    # 3-1. extract raw data sequence for each cell #
    ################################################
    # data_root = "./data/excel/time"
    # save_root = "./data/numpy/time/raw"
    # extract_cell_data(data_root, save_root, del_cells=DEL_CELLS)
    # count_data(save_root)

    ###########################################################
    # 3-2. dividing raw CAPACITY data with LOADING MASS OF AM #
    ###########################################################
    # meta_file = "./data/pickle/cell_meta_raw.pickle"
    # with open(meta_file, 'rb') as f:
    #     cell2meta = pickle.load(f)

    # data_root = "./data/numpy/time/raw"
    # save_root = "./data/numpy/time/norm"
    # DATA_ITMES_TIME = [
    #     '전류(A)', '충전 용량(Ah)', '방전 용량(Ah)', 
    #     '충전 파워(W)',  '방전 파워(W)', 
    #     '충전 에너지(Wh)', '방전 에너지(Wh)']
    # normalize_capa(cell2meta, DATA_ITMES_TIME, data_root, save_root)
    # count_data(save_root)

    #################################################
    # 3-3. split raw TIME-VOLTAGE data with 전류(A) #
    #################################################
    # save_root_cha = "./data/numpy/time/raw_split/전압(V)_충전"
    # save_root_dis = "./data/numpy/time/raw_split/전압(V)_방전"

    # data_root_current = "./data/numpy/time/norm/전류(A)"
    # data_root_voltage = "./data/numpy/time/raw/전압(V)"

    # data_list = sorted(os.listdir(data_root_current))
    
    # seq_dict = {}
    # seq_dict["cha"] = {i: [] for i in range(5)}
    # seq_dict["dis"] = {i: [] for i in range(5)}

    # # for data_idx, data_name in enumerate(tqdm(data_list)):
    # for data_idx, data_name in enumerate(data_list):
    #     cell_id = data_name[:-4]
    #     data_current = np.load(os.path.join(data_root_current, data_name))
    #     data_voltage = np.load(os.path.join(data_root_voltage, data_name))
    #     # get all indicies for cycle split
    #     # [CHARGE > DISCHARGE > CHARGE > DISCHARGE > ... ]
    #     cycle_idxs = get_idx_cycles_with_current(data_current)

    #     print("---" * 20)
    #     print(cell_id, cycle_idxs.shape)
    #     for i in range(5):
    #         cha_s, cha_e = cycle_idxs[i*2]
    #         dis_s, dis_e = cycle_idxs[i*2+1]
    #         print("... CYCLE {} | 충전 {} | 방전 {}".format(i+1, cha_e-cha_s, dis_e-dis_s))
    #         seq_dict["cha"][i].append(cha_e-cha_s)
    #         seq_dict["dis"][i].append(dis_e-dis_s)

    #     cycle_idxs_cha = cycle_idxs[[0,2,4,6,8]]
    #     for cycle_idx, (s, e) in enumerate(cycle_idxs_cha):
    #         # slice sequence
    #         draw_vol = data_voltage[s:e]
    #         # save as numpy
    #         save_dir = save_root_cha + "_cycle{}".format(cycle_idx)
    #         os.makedirs(save_dir, exist_ok=True)
    #         save_name = os.path.join(save_dir, data_name)
    #         np.save(save_name, draw_vol)
    #     cycle_idxs_dis = cycle_idxs[[1,3,5,7,9]]
    #     for cycle_idx, (s, e) in enumerate(cycle_idxs_dis):
    #         # slice sequence
    #         draw_vol = data_voltage[s:e]
    #         # save as numpy
    #         save_dir = save_root_dis + "_cycle{}".format(cycle_idx)
    #         os.makedirs(save_dir, exist_ok=True)
    #         save_name = os.path.join(save_dir, data_name)
    #         np.save(save_name, draw_vol)
    # save_root_cha = "./data/numpy/time/raw_split"
    # save_root_dis = "./data/numpy/time/raw_split"
    # count_data(save_root_cha)
    # count_data(save_root_dis)

    ##################################################################
    # 3-4. split TIME-VOLTAGE data with 전류(A) only used AVERAGEING #
    ##################################################################
    # save_root_volt_cha = "./data/numpy/time/select_split/충전_전압(V)"
    # save_root_volt_dis = "./data/numpy/time/select_split/방전_전압(V)"
    # save_root_time_cha = "./data/numpy/time/select_split/충전_시간(s)"
    # save_root_time_dis = "./data/numpy/time/select_split/방전_시간(s)"

    # data_root_current = "./data/numpy/time/norm/전류(A)"
    # data_root_voltage = "./data/numpy/time/raw/전압(V)"
    # data_root_time = "./data/numpy/time/raw/시험 시간(s)"

    # data_list = sorted(os.listdir(data_root_current))
    
    # seq_dict = {}
    # seq_dict["cha"] = {i: [] for i in range(5)}
    # seq_dict["dis"] = {i: [] for i in range(5)}
    
    # from preprocess_CYCLE_AVG_rev220306 import ids_1 as avg_list
    
    # # for data_idx, data_name in enumerate(tqdm(data_list)):
    # for data_idx, data_name in enumerate(data_list):
    #     # load TIME numpy data
    #     cell_id = data_name[:-4]
    #     data_current = np.load(os.path.join(data_root_current, data_name))
    #     data_voltage = np.load(os.path.join(data_root_voltage, data_name))
    #     data_time = np.load(os.path.join(data_root_time, data_name))
    #     # get all indicies for cycle split
    #     # [CHARGE > DISCHARGE > CHARGE > DISCHARGE > ... ]
    #     cycle_idxs = get_idx_cycles_with_current(data_current)
    #     cycle_idxs_cha = cycle_idxs[[0,2,4,6,8]]
    #     cycle_idxs_dis = cycle_idxs[[1,3,5,7,9]]
    #     avg_idxs = avg_list[cell_id][0]
    
    #     print("---" * 20)
    #     print(cell_id, cycle_idxs.shape)
    #     print(avg_idxs)

    #     for cycle_idx, avg_idx in enumerate(avg_idxs):
    #         # charge
    #         cha_s, cha_e = cycle_idxs_cha[avg_idx]
    #         data_volt_cha = data_voltage[cha_s:cha_e]
    #         data_time_cha = data_time[cha_s:cha_e]
    #         # discharge
    #         dis_s, dis_e = cycle_idxs_dis[avg_idx]
    #         data_volt_dis = data_voltage[dis_s:dis_e]
    #         data_time_dis = data_time[dis_s:dis_e]
    #         print("... [IDX {}] CHA {}, {} | DIS {}, {}".format(
    #             avg_idx, data_volt_cha.shape, data_time_cha.shape, 
    #             data_volt_dis.shape, data_time_dis.shape
    #         ))
    #         # save data as numpy
    #         # charge VOLTAGE & TIME
    #         save_dir = save_root_volt_cha + "_cycle{}".format(cycle_idx)
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, data_name), data_volt_cha)
    #         save_dir = save_root_time_cha + "_cycle{}".format(cycle_idx)
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, data_name), data_time_cha)
    #         # discharge VOLTAGE & TIME
    #         save_dir = save_root_volt_dis + "_cycle{}".format(cycle_idx)
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, data_name), data_volt_dis)
    #         save_dir = save_root_time_dis + "_cycle{}".format(cycle_idx)
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, data_name), data_time_dis)
    # save_root = "./data/numpy/time/select_split"
    # count_data(save_root)


    #################################################
    # 3-5. interpolate VOLTAGE(V) data with TIME(s) #
    #################################################
    # DATA_SEQ_LEN = 1000 # 500, 1000
    # DATA_TYPE = "충전" # 충전, 방전
    # CYCLE_IDX = 3
    # save_root_volt = "./data/numpy/time/split_interp/{}_전압(V)_cycle{}_len{}".format(DATA_TYPE, CYCLE_IDX, DATA_SEQ_LEN)
    # data_root_volt = "./data/numpy/time/select_split/{}_전압(V)_cycle{}".format(DATA_TYPE, CYCLE_IDX)
    # data_root_time = "./data/numpy/time/select_split/{}_시간(s)_cycle{}".format(DATA_TYPE, CYCLE_IDX)
    
    # data_list_volt = sorted(os.listdir(data_root_volt))
    # data_list_time = sorted(os.listdir(data_root_time))
    # assert len(data_list_volt) == len(data_list_time)

    # os.makedirs(save_root_volt, exist_ok=True)
    # for data_idx, data_name in enumerate(tqdm(data_list_volt)):
    #     # load data
    #     data_volt = np.load(os.path.join(data_root_volt, data_name))
    #     data_time = np.load(os.path.join(data_root_time, data_name))
    #     # interpolate data
    #     INTERPOLATOR = interp1d(data_time, data_volt, kind='linear')
    #     data_time_int = np.linspace(data_time.min(), data_time.max(), DATA_SEQ_LEN)
    #     data_volt_itp = INTERPOLATOR(data_time_int)
    #     # save data as numpy
    #     np.save(os.path.join(save_root_volt, data_name), data_volt_itp)

    count_data("./data/numpy/time/split_interp")