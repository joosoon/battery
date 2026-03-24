import numpy as np
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from loader.const import TESTSET_BUFF

def get_maes(outputs, gts):
    if len(outputs.shape) > 1: outputs = outputs.flatten()
    if len(gts.shape) > 1: gts = gts.flatten()
    maes = np.abs(outputs-gts)
    return maes

def get_mae(outputs, gts):
    if len(outputs.shape) > 1: outputs = outputs.flatten()
    if len(gts.shape) > 1: gts = gts.flatten()
    mae = np.average(np.abs(outputs-gts))
    return mae

def get_rmse(outputs, gts):
    if len(outputs.shape) > 1: outputs = outputs.flatten()
    if len(gts.shape) > 1: gts = gts.flatten()
    rmse = np.sqrt(np.mean((outputs-gts)**2)) 
    return rmse




# # experiment setting    
# csv_file = 'dataset/csv/filter.csv'
# in_keys = ['cathode_loading_density',
#            'cathode_porosity',
#            'cathode_AM_thickness',
#            'anode_loading_density',
#            'anode_porosity',
#            'anode_AM_thickness']
# out_keys = ['specific_capacity_0.1C', 
#             'specific_capacity_0.2C', 'specific_capacity_0.5C', 
#             'specific_capacity_1C', 'specific_capacity_2C', 
#             'specific_capacity_3C', 'specific_capacity_5C']
# fold_idx = 0
# split_set = 'FILTER_6FOLD_{}_SEED111'.format(fold_idx)
# test_ids = TESTSET_BUFF[split_set]

# # csv_file = 'dataset/csv/raw.csv'
# # test_ids = ['1.3-14','3.2-12','2.1-14','2.3-11','2.3-16','1.1-16','2.1-20',
# #             '3.2-4','3.3-6','3.1-3','2.2-10','2.3-12','2.2-33','1.2-12',
# #             '2.1-15','3.3-8','2.3-21','2.3-17','2.1-24','2.1-22','3.2-8',
# #             '2.1-19','3.1-1','2.3-32','1.2-2','1.1-26','1.1-6','3.2-5',
# #             '3.2-17','1.3-6','2.1-28','2.3-29','2.3-15','1.2-5','3.2-19',
# #             '3.1-16','3.3-10','1.1-15','3.1-13','1.3-28','1.2-13','3.2-20',
# #             '2.3-7','2.2-32','1.1-3','2.2-13','3.3-21','1.3-31','2.2-27','3.2-24']


# # # load data
# csv_data = pd.read_csv(csv_file)
# input_datas = []
# for key in in_keys:
#     datas = csv_data[key].to_numpy()
#     input_datas.append(datas)
# output_datas = []
# for key in out_keys:
#     datas = csv_data[key].to_numpy()
#     output_datas.append(datas)
# input_datas = np.stack(input_datas, axis=1)
# output_datas = np.stack(output_datas, axis=1)


# # split train set
# test_ids = np.array(test_ids)
# cell_ids = csv_data['cell_id'].to_numpy()
# train_cnd = [i for i, cell_id in enumerate(cell_ids) if cell_id not in test_ids]
# train_input_datas = input_datas[train_cnd].copy()
# train_output_datas = output_datas[train_cnd].copy()
# # split test set
# test_cnd = [i for i, cell_id in enumerate(cell_ids) if cell_id in test_ids]
# test_input_datas = input_datas[test_cnd].copy()
# test_output_datas = output_datas[test_cnd].copy()

# print("[DATA]")
# print("... (INPUT ) train: {} test: {}".format(train_input_datas.shape, test_input_datas.shape))
# print("... (OUTPUT) train: {} test: {}".format(train_output_datas.shape, test_output_datas.shape))

# # ML setting
# alpha = 0.001
# max_iter = 500
# model_type = 'forest'  # ridge, lasso, tree, forest
# if model_type == 'lasso':
#     model = Lasso(alpha=alpha, max_iter=max_iter)
# if model_type == 'ridge':  
#     model = Ridge(alpha=alpha, max_iter=max_iter)
# if model_type == 'tree':  
#     model = DecisionTreeRegressor(max_depth=3)
# if model_type == 'forest':  
#     model = RandomForestRegressor(n_estimators=1000,
#                                   criterion='mse',
#                                   random_state=1,
#                                   n_jobs=-1)

# # training
# model.fit(train_input_datas, train_output_datas)
# pr_train = model.predict(train_input_datas)
# pr_test  = model.predict(test_input_datas)

# # post-proc
# gt_train_flat = train_output_datas.flatten()
# pr_train_flat = pr_train.flatten()
# gt_test_flat = test_output_datas.flatten()
# pr_test_flat = pr_test.flatten()

# print("[RESULT]")
# print("... ( GT ) Train: {} {} | Test: {} {}".format(
#     train_output_datas.shape, gt_train_flat.shape,
#     test_output_datas.shape, gt_test_flat.shape,
# ))
# print("... (Pred) Train: {} {} | Test: {} {}".format(
#     pr_train.shape, pr_train_flat.shape,
#     pr_test.shape,  pr_test_flat.shape,
# ))

# # get metric
# r2_train = r2_score(gt_train_flat, pr_train_flat)
# r2_test  = r2_score(gt_test_flat, pr_test_flat)
# print("[R2  ] train: {:.3f} | test: {:.3f}".format(r2_train, r2_test))

# mae_train = mean_absolute_error(gt_train_flat, pr_train_flat)
# mae_test  = mean_absolute_error(gt_test_flat, pr_test_flat)
# print("[MAE ] train: {:.3f} | test: {:.3f}".format(mae_train, mae_test))

# rmse_train = get_rmse(gt_train_flat, pr_train_flat)
# rmse_test  = get_rmse(gt_test_flat, pr_test_flat)
# print("[RMSE] train: {:.3f} | test: {:.3f}".format(rmse_train, rmse_test))

# ###########
# # draw r2 #
# ###########
# save_name = 'tmp.png'
# plt.subplot(1, 2, 1)  
# plt.scatter(pr_train_flat, gt_train_flat)
# plt.title("Train R2: {:.3f}".format(r2_train))
# plt.xlabel("Prediction")
# plt.ylabel("Ground Truth")
# plt.subplot(1, 2, 2)  
# plt.scatter(pr_test_flat, gt_test_flat)
# plt.title("Test R2: {:.3f}".format(r2_test))
# plt.xlabel("Prediction")
# plt.ylabel("Ground Truth")
# plt.savefig(save_name, dpi=500)
# plt.clf() 

# #######################
# # inference each cell #
# #######################
# print(',model_type,alpha,max_iter,R2')
# print("Prediction,{},{},{},{}".format(model_type, alpha, max_iter, r2_test))
# print('crate', end=',')
# for test_id in test_ids:
#     print(test_id, end=',')
# print()
# crates = [key.split('_')[-1] for key in out_keys]
# num_crate = test_pr.shape[1]
# for crate_idx in range(num_crate):
#     crate = crates[crate_idx]
#     print(crate, end=',')
#     datas = test_pr[:, crate_idx]
#     for data in datas:
#         print(data, end=',')
#     print()



#####################
# 6-fold validation #
#####################

ours_file = 'tmp_6fold_result_ours.pickle'
with open(ours_file, 'rb') as f:
    ours_dict = pickle.load(f)


# experiment setting (common)
csv_file = 'dataset/csv/filter.csv'
in_keys = ['cathode_loading_density',
           'cathode_porosity',
           'cathode_AM_thickness',
           'anode_loading_density',
           'anode_porosity',
           'anode_AM_thickness']
out_keys = ['specific_capacity_0.1C', 
            'specific_capacity_0.2C', 'specific_capacity_0.5C', 
            'specific_capacity_1C', 'specific_capacity_2C', 
            'specific_capacity_3C', 'specific_capacity_5C']

def load_dataset(csv_file, test_ids, in_keys, out_keys, get_cell_ids_test=False):
    # load data
    csv_data = pd.read_csv(csv_file)
    input_datas = []
    for key in in_keys:
        datas = csv_data[key].to_numpy()
        input_datas.append(datas)
    output_datas = []
    for key in out_keys:
        datas = csv_data[key].to_numpy()
        output_datas.append(datas)
    input_datas = np.stack(input_datas, axis=1)
    output_datas = np.stack(output_datas, axis=1)
    # get train set
    test_ids = np.array(test_ids)
    cell_ids = csv_data['cell_id'].to_numpy()
    train_cnd = [i for i, cell_id in enumerate(cell_ids) if cell_id not in test_ids]
    xs_train = input_datas[train_cnd].copy()
    ys_train = output_datas[train_cnd].copy()
    # get test set
    test_cnd = [i for i, cell_id in enumerate(cell_ids) if cell_id in test_ids]
    xs_test = input_datas[test_cnd].copy()
    ys_test = output_datas[test_cnd].copy()

    # # tmp: print results of inference
    # cell_ids_test = cell_ids[test_cnd]
    # print("---" * 20)
    # print("[Cell ID]")
    # print(str(list(cell_ids_test)).replace("[","").replace("]", "").replace("'",""))
    if get_cell_ids_test:
        cell_ids_test = cell_ids[test_cnd]
        return (xs_train, ys_train), (xs_test, ys_test), cell_ids_test
    else:
        return (xs_train, ys_train), (xs_test, ys_test)
    
def fit_test_model(model, datas_train, datas_test, draw_r2=None, get_pr_test=False):
    # get data    
    xs_train, ys_train = datas_train
    xs_test, ys_test = datas_test
    # fitting
    model.fit(xs_train, ys_train)
    # inference
    pr_train = model.predict(xs_train)
    pr_test  = model.predict(xs_test)
    # post-proc
    gt_train_flat = ys_train.flatten()
    pr_train_flat = pr_train.flatten()
    gt_test_flat  = ys_test.flatten()
    pr_test_flat  = pr_test.flatten()
    # get metric
    r2_train   = r2_score(gt_train_flat, pr_train_flat)
    r2_test    = r2_score(gt_test_flat, pr_test_flat)
    mae_train  = get_mae(gt_train_flat, pr_train_flat)
    mae_test   = get_mae(gt_test_flat, pr_test_flat)
    rmse_train = get_rmse(gt_train_flat, pr_train_flat)
    rmse_test  = get_rmse(gt_test_flat, pr_test_flat)
    if draw_r2 is not None:
        save_name = draw_r2
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(gt_test_flat, pr_test_flat)
        xs = np.arange(gt_test_flat.min(), gt_test_flat.max(), (gt_test_flat.max()-gt_test_flat.min())/500)
        plt.scatter(xs, xs, s=0.15, c='black')
        plt.title("Test R2: {:.3f}".format(r2_test))
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.savefig(save_name, dpi=500)
        plt.clf() 
        # print("... saved at", save_name)


        # tmp: print results of inference
        # print("---" * 20)
        # print("[GT]")
        # # print(ys_test)
        # ys_test = np.transpose(ys_test)
        # for i in ys_test:
        #     print(str(list(i)).replace("[","").replace("]", ""))
        # print("---" * 20)
        # print("[Pred]")
        # print(np.round(pr_test, 4))
        # pr_test = np.transpose(pr_test)
        # for i in pr_test:
        #     print(str(list(i)).replace("[","").replace("]", ""))
        # exit()
        
        pr_test = np.transpose(pr_test)
        for i in pr_test:
            print(str(list(i)).replace("[","").replace("]", ""))
    if get_pr_test:
        return (r2_train, r2_test), (mae_train, mae_test), (rmse_train, rmse_test), (ys_test, pr_test)
    else:
        return (r2_train, r2_test), (mae_train, mae_test), (rmse_train, rmse_test)
    
def run(model, model_type, key):
    for fold_idx in range(6):
        split_set = 'FILTER_6FOLD_{}_SEED111'.format(fold_idx)
        test_ids = TESTSET_BUFF[split_set]
        datas_train, datas_test = load_dataset(csv_file, test_ids, in_keys, out_keys)
        r2, mae, rmse = fit_test_model(model, datas_train, datas_test)
        r2_train, r2_test     = r2
        mae_train, mae_test   = mae
        rmse_train, rmse_test = rmse
        # print result
        print("{},fold_{},{},{},{},{},{},{},{}".format(
              model_type, fold_idx, key, r2_train, r2_test, 
              mae_train, mae_test, rmse_train, rmse_test))


# for printing results of inference
def run_fold_prt(model, model_type, key, fold_idx):
    # set testset
    split_set = 'FILTER_6FOLD_{}_SEED111'.format(fold_idx)
    split_set = 'VIS_TESTSET'
    test_ids = TESTSET_BUFF[split_set]
    datas_train, datas_test = load_dataset(csv_file, test_ids, in_keys, out_keys)
    r2, mae, rmse = fit_test_model(model, datas_train, datas_test, draw_r2="gt-pred_{}.png".format(model_type))
    r2_train, r2_test     = r2
    mae_train, mae_test   = mae
    rmse_train, rmse_test = rmse
    # # print result
    # print("{},fold_{},{},{},{},{},{},{},{}".format(
    #         model_type, fold_idx, key, r2_train, r2_test, 
    #         mae_train, mae_test, rmse_train, rmse_test))

def draw_all_cell_6fold(model, model_type):
    gt_all, pr_all = [], []
    gts, prs = [], []
    cell_all = []
    fold_idx = 0
    for fold_idx in range(6):
        split_set = 'FILTER_6FOLD_{}_SEED111'.format(fold_idx)
        test_ids = TESTSET_BUFF[split_set]
        datas_train, datas_test, cell_ids_test = load_dataset(csv_file, test_ids, in_keys, out_keys, get_cell_ids_test=True)
        r2, mae, rmse, gt_pr = fit_test_model(model, datas_train, datas_test, get_pr_test=True)
        gt_test, pr_test = gt_pr
        gt_flat = gt_test.flatten()
        pr_flat = pr_test.flatten()
        gt_all.append(gt_flat)
        pr_all.append(pr_flat)

        # keep data for print
        cell_all.append(cell_ids_test)
        gts.append(gt_test)
        prs.append(pr_test)
    
    gt_all = np.concatenate(gt_all)
    pr_all = np.concatenate(pr_all)
    # draw
    save_name = 'tmp_r2_{}.png'.format(model_type)
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(gt_all, pr_all, s=15, c='white', edgecolors='green')
    # tmp: draw ours together
    plt.scatter(ours_dict['gt'], ours_dict['pr'], s=15, c='white', edgecolors='blue')
    min_val = -10 # gt_all.min() #   
    max_val = 200 # gt_all.max() #   
    xs = np.arange(min_val, max_val, (max_val-min_val)/500)
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.scatter(xs, xs, s=0.15, c='black')
    plt.title("Test R2: {:.5f}".format(r2_test))
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.savefig(save_name, dpi=500)
    plt.clf() 
    # print("... saved at", save_name)

    # data for print
    cell_all = np.concatenate(cell_all).transpose()
    gts = np.concatenate(gts).transpose()
    prs = np.concatenate(prs).transpose()

    # sort by cell-ids
    import pandas as pd
    prt_dict = {}
    prt_dict['cell_id'] = cell_all
    prt_dict['ids'] = np.arange(len(cell_all))
    prt_df = pd.DataFrame(prt_dict).sort_values('cell_id')
    ids = prt_df['ids'].to_numpy()
    gts = gts[:, ids]
    prs = prs[:, ids]
    
    # print("---" * 20)
    # print(str(list(cell_all[ids])).replace("[","").replace("]", "").replace("'",""))
    # print("---" * 20)
    # for i in gts:
    #     print(str(list(i)).replace("[","").replace("]", ""))
    print("---" * 20)
    for i in prs:
        print(str(list(i)).replace("[","").replace("]", ""))


##############################################
# draw R2 of all testsets of 6-fold together #
##############################################



model_type = 'ridge'
print(model_type)
alpha = 1
model = Ridge(alpha=alpha, max_iter=500)
draw_all_cell_6fold(model, model_type)

model_type = 'lasso'
print(model_type)
alpha = 0.01
model = Lasso(alpha=alpha, max_iter=500)
draw_all_cell_6fold(model, model_type)

model_type = 'elastic'
print(model_type)
alpha = 0.01
l1_ratio = 0.9
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
draw_all_cell_6fold(model, model_type)

model_type = 'tree'
print(model_type)
max_depth = 8
model = DecisionTreeRegressor(max_depth=max_depth)
draw_all_cell_6fold(model, model_type)

model_type = 'forest'
print(model_type)
n_estimators = 10
random_state = 999
criterion = 'mae'
model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            random_state=random_state,
            n_jobs=-1)
draw_all_cell_6fold(model, model_type)



exit()

##############################################
# print the result of inference at each fold #
##############################################
fold_idx = 0
key = "tmp"

model_type = 'ridge'
print(model_type)
alpha = 1
model = Ridge(alpha=alpha, max_iter=500)
run_fold_prt(model, model_type, key, fold_idx)

model_type = 'lasso'
print(model_type)
alpha = 0.01
model = Lasso(alpha=alpha, max_iter=500)
run_fold_prt(model, model_type, key, fold_idx)

model_type = 'elastic'
print(model_type)
alpha = 0.01
l1_ratio = 0.9
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
run_fold_prt(model, model_type, key, fold_idx)

model_type = 'tree'
print(model_type)
max_depth = 8
model = DecisionTreeRegressor(max_depth=max_depth)
run_fold_prt(model, model_type, key, fold_idx)

model_type = 'forest'
print(model_type)
n_estimators = 10
random_state = 999
criterion = 'mae'
model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            random_state=random_state,
            n_jobs=-1)
run_fold_prt(model, model_type, key, fold_idx)
exit()



#################################
# print average metric of model #
#################################
# Ridge 
model_type = 'ridge'
for alpha in [1, 0.5, 0.1, 0.05, 0.01]:
    model = Ridge(alpha=alpha, max_iter=500)
    key = "{}".format(alpha)
    run(model, model_type, key)

# Lasso
model_type = 'lasso'
for alpha in [1, 0.5, 0.1, 0.05, 0.01]:
    model = Lasso(alpha=alpha, max_iter=500)
    key = "{}".format(alpha)
    run(model, model_type, key)
    
# Elastic
model_type = 'elastic'
for alpha in [1, 0.5, 0.1, 0.05, 0.01]:
    for l1_ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        key = "{}-{}".format(alpha, l1_ratio)
        run(model, model_type, key)

# Decision tree
model_type = 'tree'
for max_depth in [1, 2, 3, 5, 6, 7, 8, 9, 10]:
    model = DecisionTreeRegressor(max_depth=max_depth)
    key = "{}".format(max_depth)
    run(model, model_type, key)

# Random forest
model_type = 'forest'
for n_estimators in [10, 100, 1000, 5000]:
    for random_state in [111, 333, 555, 777, 999]:
        for criterion in ['mse', 'mae']:
            model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        criterion=criterion,
                        random_state=random_state,
                        n_jobs=-1)
            key = "{}-{}-{}".format(n_estimators, random_state, criterion)
            run(model, model_type, key)

exit()
