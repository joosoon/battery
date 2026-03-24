## load libraries
import smogn
import pandas
# import ImbalancedLearningRegression as iblr
import sys
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

from loader.const import TESTSET_BUFF


DATASEED = 111
for FOLD_IDX in range(6):
    test_cell_ids = TESTSET_BUFF['FILTER_6FOLD_{}_SEED{}'.format(FOLD_IDX, DATASEED)]
    save_root = 'dataset/csv_smote_seed_{}/fold_{}'.format(DATASEED, FOLD_IDX)
    
    # tvt
    test_cell_ids = TESTSET_BUFF['FILTER_6FOLD_{}_SEED{}'.format(FOLD_IDX, DATASEED)] \
                  + TESTSET_BUFF['FILTER_6FOLD_{}_SEED{}'.format(5-FOLD_IDX, DATASEED)] 
    save_root = 'dataset/csv_smote_seed_{}_tvt/fold_{}'.format(DATASEED, FOLD_IDX)
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, 'raw')
    os.makedirs(save_dir, exist_ok=True)

    ##################################
    # train-test split from raw data # 
    ##################################
    # load data
    csv_file = "dataset/csv/raw.csv"
    csv_raw = pandas.read_csv(csv_file)
    # get train-test split ids 
    cell_ids = csv_raw['cell_id'].to_list()
    test_ids = [cell_ids.index(cell_id) for cell_id in test_cell_ids]
    test_cnd = np.zeros(len(cell_ids), dtype=np.bool_)
    test_cnd[test_ids] = True
    # split with ids
    def split_by_cnd(data_dict, cnd):
        new_dict = {}
        for col in data_dict.columns:
            new_dict[col] = data_dict[col].to_numpy()[cnd]
        return pd.DataFrame(new_dict)
    csv_train = split_by_cnd(csv_raw, ~test_cnd)
    csv_test = split_by_cnd(csv_raw, test_cnd)
    # save as csv
    csv_train.to_csv(os.path.join(save_root, "org_train.csv"))
    csv_test.to_csv(os.path.join(save_root, "org_test.csv"))

    ##################################
    # SMOTE augmentation on trainset #
    ##################################
    # load trainset
    csv_file = os.path.join(save_root, "org_train.csv")
    csv_raw = pandas.read_csv(csv_file)
    # smote
    ##############################################################
    # [rel_coef] coefficient for box plot (pos real)             #
    # [rel_thres] relevance threshold considered rare (pos real) #
    # [k] num of neighs for over-sampling (pos int)              #
    # [pert] perturbation / noise percentage (pos real)          #
    ##############################################################
    params = {}
    params['num_neih'] = [3, 5, 7, 9, 13]
    params['rel_coef'] = [0.01, 0.1, 1.0, 1.5, 3.0]
    params['rel_thre'] = [0.1, 0.2, 0.5, 0.7, 1.0]
    params['perturb']  = [0.01, 0.05, 0.1, 0.25, 0.5]

    params['num_neih'] = [3, 5, 9]
    params['rel_coef'] = [0.01, 1.5, 3.0]
    params['rel_thre'] = [0.2, 0.5, 0.7]
    params['perturb']  = [0.02, 0.15, 0.5]
    param_list = []
    for k in params['num_neih']:
        for rel_coef in params['rel_coef']:
            for rel_thres in params['rel_thre']:
                for pert in params['perturb']:
                    param = [k, rel_coef, rel_thres, pert]
                    param_list.append(param)
    np.random.shuffle(param_list)

    targets = [
            'cathode_loading_density',
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
            'specific_capacity_5C',
        ]
    for target_idx, smote_target in enumerate(targets):
        print("[{}/{}] {}".format(target_idx+1, len(targets), smote_target))
        for param in tqdm(param_list):
            k, rel_coef, rel_thres, pert = param
            try:
                csv_smote = smogn.smoter(
                            data=csv_raw, 
                            y=smote_target,
                            pert=pert, k=k,
                            rel_coef=rel_coef, 
                            rel_thres=rel_thres, 
                        )
                # filtering outlier
                save_name = smote_target + "_" + "_".join(map(str,param)) + '.csv'
                save_name = os.path.join(save_dir, save_name)
                csv_smote.to_csv(save_name)
            except:
                print("... failed ", param)
