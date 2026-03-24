
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt




######################
# sum and filter csv #
######################
TARGETS = ['cathode_loading_density',
           'cathode_porosity',
           'cathode_AM_thickness',
           'anode_loading_density',
           'anode_porosity',
           'anode_AM_thickness',
           'specific_capacity_0.1C',
           'specific_capacity_0.2C',
           'specific_capacity_0.5C',
           'specific_capacity_1C',
           'specific_capacity_2C',
           'specific_capacity_3C',
           'specific_capacity_5C',]

DATASEED = 111
for fold_idx in range(6):
    smt_root = 'dataset/csv_smote_seed_{}_tvt/fold_{}'.format(DATASEED, fold_idx)
    # load raw data
    raw_file = os.path.join(smt_root, 'org_train.csv')
    raw_csv = pd.read_csv(raw_file)
    # prevent too similar values
    round_digit = 3
    # fileter ver.1: all items are same #
    # get raw datas
    data_strs = []
    for i in range(len(raw_csv)):
        data_str = []
        for target in TARGETS:
            data = raw_csv[target][i].round(round_digit)
            data_str.append(data)
        # # filter ver.1: string, whole same
        # data_str = list(map(str, data_str))
        # data_str = "-".join(data_str)
        # filter ver.2: numpy, over 3 items are same
        data_str = np.array(data_str)
        data_strs.append(data_str)
    data_strs = np.stack(data_strs)
    # load smote data (concat)
    for smt_target in tqdm(TARGETS):
        smt_dir   = os.path.join(smt_root, 'raw')
        
        # # filter ver.1: string, whole same
        # save_root = os.path.join(smt_root, 'filter')
        # filter ver.2: numpy, over 3 items are same
        save_root = os.path.join(smt_root, 'filter_v2')
        os.makedirs(save_root, exist_ok=True)
        # sum csv of smote target
        smt_list = [name for name in os.listdir(smt_dir) 
                    if name.startswith(smt_target)]
        smt_csv = None
        for smt_name in smt_list:
            smt_file = os.path.join(smt_dir, smt_name) 
            smt_csv_tmp = pd.read_csv(smt_file)
            if smt_csv is None:
                smt_csv = smt_csv_tmp
            else:
                smt_csv = pd.concat([smt_csv, smt_csv_tmp], ignore_index=True)
        # filter similar datas
        filtered_ids = []
        for i in tqdm(range(len(smt_csv))):
            data_str = []
            for target in TARGETS:
                data = smt_csv[target][i].round(round_digit)
                data_str.append(data)
            
            # # filter ver.1: string, whole same
            # data_str = list(map(str, data_str))
            # data_str = "-".join(data_str)
            # if data_str not in data_strs:
                # data_strs.append(data_str)
                # filtered_ids.append(i)

            # filter ver.2: numpy, over 3 items are same
            thres_filter = 3
            data_str = np.array(data_str)
            cnd = data_strs[:, :]==data_str
            cnd = cnd.sum(axis=1)
            cnd = cnd > thres_filter
            cnd = cnd.sum()
            if cnd == 0:
                data_str = np.expand_dims(data_str, axis=0)
                data_strs = np.concatenate((data_strs, data_str), axis=0)
                filtered_ids.append(i)
        # save filtered datas as csv 
        save_dict = {}
        save_dict['cell_id'] = ["{}-{}".format(smt_target, i)
                                for i in range(len(filtered_ids))]
        for target in TARGETS: 
            save_dict[target] = []
        for i in filtered_ids:
            for target in TARGETS:
                data = smt_csv[target][i]
                save_dict[target].append(data)
        save_df = pd.DataFrame(save_dict)
        save_file = os.path.join(save_root, "{}.csv".format(smt_target))
        save_df.to_csv(save_file)
        print("[{}] {} datas".format(smt_target, len(filtered_ids)))
        print("... saved at", save_file)
        exit()

exit()

###################
# draw smote data #
###################
TARGETS = ['cathode_loading_density',
           'cathode_porosity',
           'cathode_AM_thickness',
           'anode_loading_density',
           'anode_porosity',
           'anode_AM_thickness',
           'specific_capacity_0.1C',
           'specific_capacity_0.2C',
           'specific_capacity_0.5C',
           'specific_capacity_1C',
           'specific_capacity_2C',
           'specific_capacity_3C',
           'specific_capacity_5C',]

axis_dict = []
axis_dict.append({'x': 'cathode_loading_density',
                  'y': 'cathode_porosity'})
axis_dict.append({'x': 'anode_loading_density',
                  'y': 'anode_porosity'})
axis_dict.append({'x': 'cathode_porosity',
                  'y': 'anode_porosity'})
axis_dict.append({'x': 'cathode_loading_density',
                  'y': 'specific_capacity_0.1C'})
axis_dict.append({'x': 'cathode_loading_density',
                  'y': 'specific_capacity_1C'})
axis_dict.append({'x': 'cathode_loading_density',
                  'y': 'specific_capacity_5C'})

DATASEED = 111
for fold_idx in range(6):
    raw_root = 'dataset/csv_smote_seed_{}_tvt/fold_{}'.format(DATASEED, fold_idx)
    smt_root = os.path.join(raw_root, 'filter')
    vis_root = os.path.join(raw_root, 'vis')
    os.makedirs(vis_root, exist_ok=True)
    # load data
    raw_file = os.path.join(raw_root, 'org_train.csv')
    raw_csv = pd.read_csv(raw_file)
    
    sum_csv = None
    for smt_name in sorted(os.listdir(smt_root)):
        smt_target = smt_name.replace(".csv", "")
        save_file = os.path.join(vis_root, "{}.png".format(smt_target))
        # load data
        smt_file = os.path.join(smt_root, smt_name)
        smt_csv = pd.read_csv(smt_file)
        # sum csv
        sum_csv = smt_csv if sum_csv is None else pd.concat([sum_csv, smt_csv], ignore_index=True)
        # draw each smote
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle('SMOTE with [{}] : {} datas'.format(smt_target, len(smt_csv)))
        for axis_i, axis in enumerate(axis_dict):
            axis_x = axis['x']
            axis_y = axis['y']
            raw_x = raw_csv[axis_x].to_list()
            raw_y = raw_csv[axis_y].to_list()
            smt_x = smt_csv[axis_x].to_list()
            smt_y = smt_csv[axis_y].to_list()

            plt.subplot(2, 3, axis_i+1)
            plt.scatter(raw_x, raw_y, label='Raw')
            plt.scatter(smt_x, smt_y, label='SMOTE', alpha=0.25)
            plt.legend(loc='best')
            plt.xlabel(axis_x)
            plt.ylabel(axis_y)
        plt.savefig(save_file, dpi=500)
        plt.clf()
        print("... saved at", save_file)
    # draw sum csv
    save_file = os.path.join(vis_root, "all.png")
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle('SMOTE with [{}] : {} datas'.format(smt_target, len(smt_csv)))
    for axis_i, axis in enumerate(axis_dict):
        axis_x = axis['x']
        axis_y = axis['y']
        raw_x = raw_csv[axis_x].to_list()
        raw_y = raw_csv[axis_y].to_list()
        smt_x = sum_csv[axis_x].to_list()
        smt_y = sum_csv[axis_y].to_list()
        plt.subplot(2, 3, axis_i+1)
        plt.scatter(raw_x, raw_y, label='Raw')
        plt.scatter(smt_x, smt_y, label='SMOTE', alpha=0.25)
        plt.legend(loc='best')
        plt.xlabel(axis_x)
        plt.ylabel(axis_y)
    plt.savefig(save_file, dpi=500)
    plt.clf()
    print("... saved at", save_file)    

exit()
