from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import numpy as np
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
csv_file = 'dataset/csv/raw.csv'
in_keys = [        
        'cathode_loading_density',
        'cathode_porosity',
        'cathode_AM_thickness',
        'anode_loading_density',
        'anode_porosity',
        'anode_AM_thickness',
        ]
out_keys = ['specific_capacity_0.1C', 
    'specific_capacity_0.2C', 'specific_capacity_0.5C', 
    'specific_capacity_1C', 'specific_capacity_2C', 
    'specific_capacity_3C', 'specific_capacity_5C']
from loader.batterycell import TransformCell
transform = TransformCell([])

# load data
csv_data = pd.read_csv(csv_file)
input_datas = []
for key in in_keys:
    datas = csv_data[key].to_numpy()
    # datas /= transform.norm_vals[key]
    input_datas.append(datas)
input_datas = np.stack(input_datas, axis=1)

output_datas = []
for key in out_keys:
    datas = csv_data[key].to_numpy()
    # datas /= transform.norm_vals[key]
    output_datas.append(datas)
output_datas = np.stack(output_datas, axis=1)

cell_ids = csv_data['cell_id'].to_numpy()
test_ids = ['1.3-14','3.2-12','2.1-14','2.3-11','2.3-16','1.1-16','2.1-20',
            '3.2-4','3.3-6','3.1-3','2.2-10','2.3-12','2.2-33','1.2-12',
            '2.1-15','3.3-8','2.3-21','2.3-17','2.1-24','2.1-22','3.2-8',
            '2.1-19','3.1-1','2.3-32','1.2-2','1.1-26','1.1-6','3.2-5',
            '3.2-17','1.3-6','2.1-28','2.3-29','2.3-15','1.2-5','3.2-19',
            '3.1-16','3.3-10','1.1-15','3.1-13','1.3-28','1.2-13','3.2-20',
            '2.3-7','2.2-32','1.1-3','2.2-13','3.3-21','1.3-31','2.2-27','3.2-24']
test_ids = np.array(test_ids)
test_cnd = [i for i, cell_id in enumerate(cell_ids) if cell_id in test_ids]
train_cnd = [i for i, cell_id in enumerate(cell_ids) if cell_id not in test_ids]

print("... INPUT DATA :", input_datas.shape)
print("... OUTPUT DATA:", output_datas.shape)

train_input_datas = input_datas[train_cnd].copy()
train_output_datas = output_datas[train_cnd].copy()
test_input_datas = input_datas[test_cnd].copy()
test_output_datas = output_datas[test_cnd].copy()

print("...", train_input_datas.shape)
print("...", train_output_datas.shape)
print("...", test_input_datas.shape)
print("...", test_output_datas.shape)



alpha = 0.001
max_iter = 500
model_type = 'ridge'

if model_type == 'lasso':
    model = Lasso(alpha=alpha, max_iter=max_iter)
if model_type == 'ridge':  
    model = Ridge(alpha=alpha, max_iter=max_iter)
model.fit(train_input_datas, train_output_datas)
train_pr = model.predict(train_input_datas)
test_pr = model.predict(test_input_datas)

gt_train_flat = train_output_datas.flatten()
pr_train_flat = train_pr.flatten()
r2_train  = r2_score(gt_train_flat, pr_train_flat)

gt_test_flat = test_output_datas.flatten()
pr_test_flat = test_pr.flatten()
r2_test  = r2_score(gt_test_flat, pr_test_flat)


print(',model_type,alpha,max_iter,R2')
print("Prediction,{},{},{},{}".format(model_type, alpha, max_iter, r2_test))
print('crate', end=',')
for test_id in test_ids:
    print(test_id, end=',')
print()

crates = [key.split('_')[-1] for key in out_keys]
num_crate = test_pr.shape[1]
for crate_idx in range(num_crate):
    crate = crates[crate_idx]
    print(crate, end=',')
    datas = test_pr[:, crate_idx]
    for data in datas:
        print(data, end=',')
    print()


exit()

save_root = 'result_ml'
os.makedirs(save_root, exist_ok=True)

print("---" * 20)
print("model type,alpha,max iter,train R2,test R2")

for model_type in ['ridge', 'lasso']:
    for alpha in [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        for max_iter in [10, 100, 1000, 10000, 100000]:
            if model_type == 'lasso':
                model = Lasso(alpha=alpha, max_iter=max_iter)
            if model_type == 'ridge':  
                model = Ridge(alpha=alpha, max_iter=max_iter)
            model.fit(train_input_datas, train_output_datas)
            train_pr = model.predict(train_input_datas)
            test_pr = model.predict(test_input_datas)

            gt_train_flat = train_output_datas.flatten()
            pr_train_flat = train_pr.flatten()
            r2_train  = r2_score(gt_train_flat, pr_train_flat)

            gt_test_flat = test_output_datas.flatten()
            pr_test_flat = test_pr.flatten()
            r2_test  = r2_score(gt_test_flat, pr_test_flat)

            print("{},{},{},{},{}".format(model_type, alpha, max_iter, r2_train, r2_test))
            # draw R2
            save_name = "{}_{}_{}.png".format(model_type, alpha, max_iter)
            save_name = os.path.join(save_root, save_name)

            plt.subplot(1, 2, 1)  
            plt.scatter(pr_train_flat, gt_train_flat)
            plt.title("Train R2: {:.3f}".format(r2_train))
            plt.xlabel("Prediction")
            plt.ylabel("Ground Truth")
            plt.subplot(1, 2, 2)  
            plt.scatter(pr_test_flat, gt_test_flat)
            plt.title("Test R2: {:.3f}".format(r2_test))
            plt.xlabel("Prediction")
            plt.ylabel("Ground Truth")
            plt.savefig(save_name, dpi=500)
            plt.clf() 




exit()


kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp = GaussianProcessRegressor()
gp.fit(train_input_datas, train_output_datas)
train_pr, sigma = gp.predict(train_input_datas, return_std=True)
test_pr, sigma = gp.predict(test_input_datas, return_std=True)

print(train_pr.shape)
print(test_pr.shape)

train_r2 = r2_score(train_output_datas, train_pr)
test_r2  = r2_score(test_output_datas, test_pr)


print("---" * 20)
print("GAUSSIAN PROCESS")
print(train_r2)
print(test_r2)

