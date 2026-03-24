## load libraries
import smogn
import pandas
import ImbalancedLearningRegression as iblr
import sys
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')



xs_name = 'cathode_loading_density'
ys_name = 'cathode_porosity'
cs_name = 'specific_capacity_0.1C'

data_root = 'dataset/csv_smote'
data_list = os.listdir(data_root)
np.random.shuffle(data_list)

fig = plt.figure(figsize=(35, 5))
cap_names = ['specific_capacity_0.1C',
             'specific_capacity_0.2C',
             'specific_capacity_0.5C',
             'specific_capacity_1C',
             'specific_capacity_2C',
             'specific_capacity_3C',
             'specific_capacity_5C',
            ]

for idx, cap_name in enumerate(cap_names):
    print(cap_name)
    plt.subplot(1, len(cap_names), idx+1)
    for data_idx, data_name in enumerate(data_list[:50]):
        smote_key = "_".join(data_name.split("_")[:-4])
        data_file = os.path.join(data_root, data_name)
        data_csv = pandas.read_csv(data_file)
        xs = data_csv[xs_name]
        ys = data_csv[ys_name]
        cs = data_csv[cs_name]
        plt.scatter(xs, ys, color='gray', label=smote_key, alpha=0.4)
        
    raw_file = "dataset/csv/raw_train.csv"
    raw_csv = pandas.read_csv(data_file)
    xs = raw_csv[xs_name]
    ys = raw_csv[ys_name]
    cs = raw_csv[cs_name]
    plt.scatter(xs, ys, color='red', label='Raw'.format(data_idx), alpha=1.0)
save_name = 'tmp_smote.png'
plt.savefig(save_name, dpi=500)
plt.clf()
print("... saved at", save_name)

exit()


##################################
# train-test split from raw data # 
##################################
test_cell_ids = [
    '1.1-3', '1.1-6', '1.1-15', '1.1-16', '1.1-26', '1.2-2', '1.2-5', '1.2-12', '1.2-13', '1.3-6', 
    '1.3-14', '1.3-28', '1.3-31', '2.1-14', '2.1-15', '2.1-19', '2.1-20', '2.1-22', '2.1-24', '2.1-28', 
    '2.2-10', '2.2-13', '2.2-27', '2.2-32', '2.2-33', '2.3-7', '2.3-11', '2.3-12', '2.3-15', '2.3-16', 
    '2.3-17', '2.3-21', '2.3-29', '2.3-32', '3.1-1', '3.1-3', '3.1-13', '3.1-16', '3.2-4', '3.2-5', 
    '3.2-8', '3.2-12', '3.2-17', '3.2-19', '3.2-20', '3.2-24', '3.3-6', '3.3-8', '3.3-10', '3.3-21', 
    ]

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
csv_train.to_csv("dataset/csv/raw_train.csv")
csv_test.to_csv("dataset/csv/raw_test.csv")


##################################
# SMOTE augmentation on trainset #
##################################
save_root = 'dataset/csv_smote'
os.makedirs(save_root, exist_ok=True)
# load trainset
csv_file = "dataset/csv/raw_train.csv"
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
# for target_idx, smote_target in enumerate(targets):
#     print("[{}/{}] {}".format(target_idx+1, len(targets), smote_target))
#     for param in tqdm(param_list):
#         k, rel_coef, rel_thres, pert = param
#         try:
#             csv_smote = smogn.smoter(
#                         data=csv_raw, 
#                         y=smote_target,
#                         pert=pert, k=k,
#                         rel_coef=rel_coef, 
#                         rel_thres=rel_thres, 
#                     )
#             # filtering outlier 
#             save_name = smote_target + "_" + "_".join(map(str,param))
#             save_name = os.path.join(save_root, save_name)
#             csv_smote.to_csv(save_name)
#         except:
#             print("... failed ", param)

key2num_datas = {}
key2num_smote = {}
data_root = 'dataset/csv_smote'
data_list = sorted(os.listdir(data_root))
for data_idx, data_name in enumerate(data_list):
    data_file = os.path.join(data_root, data_name)
    data_csv = pandas.read_csv(data_file)
    num_data = len(data_csv)
    smote_key = "_".join(data_name.split("_")[:-4])
    if smote_key not in key2num_smote: key2num_smote[smote_key] = []
    if smote_key not in key2num_datas: key2num_datas[smote_key] = 0
    
    key2num_smote[smote_key].append(data_name)
    key2num_datas[smote_key] += num_data
    
num_data_all = 0
for k in key2num_smote.keys():
    num_smote = len(key2num_smote[k])
    num_datas = key2num_datas[k]
    num_data_all += num_datas
    print("[{:<23s}] {} smote | {} datas".format(k, num_smote, num_datas))
print("---" * 20)
print("total: {}".format(num_data_all))
exit()

num_data_all = 0
data_root = 'dataset/csv_smote'
data_list = sorted(os.listdir(data_root))
for data_name in data_list:
    data_file = os.path.join(data_root, data_name)
    data_csv = pandas.read_csv(data_file)
    print(data_name, len(data_csv))
    num_data = len(data_csv)
    num_data_all += num_data
print("---" * 20)
print("total: {}".format(num_data_all))

xs_name = 'cathode_loading_density'
ys_name = 'cathode_porosity'
cs_name = 'specific_capacity_0.1C'

data_root = 'dataset/csv_smote'
data_list = os.listdir(data_root)
for data_idx, data_name in enumerate(data_list):
    smote_key = "_".join(data_name.split("_")[:-4])
    data_file = os.path.join(data_root, data_name)
    data_csv = pandas.read_csv(data_file)
    xs = data_csv[xs_name]
    ys = data_csv[ys_name]
    cs = data_csv[cs_name]
    plt.scatter(xs, ys, color='gray', label=smote_key, alpha=0.2)
    
raw_file = "dataset/csv/raw_train.csv"
raw_csv = pandas.read_csv(data_file)
xs = raw_csv[xs_name]
ys = raw_csv[ys_name]
cs = raw_csv[cs_name]
plt.scatter(xs, ys, color='red', label='Raw'.format(data_idx), alpha=1.0)

plt.legend(loc='best')

save_name = 'tmp_smote.png'
plt.savefig(save_name, dpi=500)
plt.clf()
print("... saved at", save_name)

exit()




####################################
# filtering outlier with threshold #
####################################



def draw_capacity_graph(data_dict, save_name='tmp.png'):
    xs = data_dict['cathode_loading_density']
    ys = data_dict['cathode_porosity']
    
    fig = plt.figure(figsize=(35, 5))
    cap_names = ['specific_capacity_0.1C',
                 'specific_capacity_0.2C',
                 'specific_capacity_0.5C',
                 'specific_capacity_1C',
                 'specific_capacity_2C',
                 'specific_capacity_3C',
                 'specific_capacity_5C',
                ]
    for idx, cap_name in enumerate(cap_names):
        cap_data = data_dict[cap_name]
        print(cap_name)
        plt.subplot(1, len(cap_names), idx+1)
        plt.scatter(xs, ys, c=cap_data)
        plt.title(cap_name)
        plt.colorbar()
    if not save_name.endswith(".png"):
        save_name += ".png"
    plt.savefig(save_name)
    plt.clf()
    print("... saved at", save_name)
        
        
        
        
## load data
csv_file = "dataset/csv/raw.csv"
csv_data = pandas.read_csv(csv_file)
csv_data = csv_data.drop(columns=['cell_id'])
names = ['origin']
datas = [csv_data]

# draw_capacity_graph(csv_data, 'raw')

names.append("0.1C")
data_smogn = smogn.smoter(
    data = csv_data,
    y = 'cathode_loading_density',
    rel_coef = 0.01,
    k = 9,
    rel_thres = 0.5,
)
draw_capacity_graph(data_smogn, 'c_loading')


names.append("0.1C")
data_smogn_2 = smogn.smoter(
    data = csv_data,
    y = 'specific_capacity_0.1C',
    k = 9,
    rel_thres = 0.5,
)
draw_capacity_graph(data_smogn_2, 'smopte_0.1C')


exit()

datas.append(data_smogn)

names.append("C_Loading")
data_smogn = smogn.smoter(
    data = csv_data, 
    y = 'cathode_loading_density', 
    rel_coef = 0.01,
    k = 9,
    rel_thres = 0.5,          ## relevance threshold considered rare (pos real)
)
datas.append(data_smogn)

names.append("C_Porosity")
data_smogn = smogn.smoter(
    data = csv_data, 
    y = 'cathode_porosity', 
    rel_coef = 0.01,
    k = 9,
    rel_thres = 0.5,          ## relevance threshold considered rare (pos real)
)
datas.append(data_smogn)


# draw 
keys = [
    'cathode_loading_density',
    'cathode_porosity',
    'anode_loading_density',
    'anode_porosity',
    'specific_capacity_0.1C',
    'specific_capacity_0.2C',
    'specific_capacity_3C',
    'specific_capacity_5C',
        ]
# fig = plt.figure(figsize=(20, 30))
fig = plt.figure(figsize=(25, 5))
# for data_i, (data, name) in enumerate(zip(datas, names)):
for data_i, (data, name) in enumerate(zip(reversed(datas), reversed(names))):
    for key_idx, key in enumerate(keys):
        # plt.subplot(len(datas), len(keys), data_i+key_idx+2)
        plt.subplot(1, len(keys), key_idx+1)
        plt.hist(data[key], 10, label=name) # histtype='step')
        plt.legend(loc='best')
        plt.title(key)
save_name = "tmp.png"
plt.savefig(save_name)
plt.clf()
print("... saved at", save_name)
exit()

# draw 
keys = [
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
for data_i, (data, name) in enumerate(zip(datas, names)):
    fig = plt.figure(figsize=(20, 30))
    for key_idx, key in enumerate(keys):
        plt.subplot(2, 7, key_idx+1)
        plt.hist(data[key], 10)
        plt.title(key)
    plt.suptitle(name)
    plt.savefig("tmp_{}.png".format(name))
    plt.clf()


# draw 
fig = plt.figure(figsize=(16, 10))
x_key = 'cathode_loading_density'
y_key = 'cathode_porosity'
z_key = 'specific_capacity_0.1C'
for data_i, (data, name) in enumerate(zip(datas, names)):
    xs = data[x_key]
    ys = data[y_key]
    zs = data[z_key]

    # delete outlier
    
    plt.subplot(1, len(names), data_i+1)
    plt.scatter(xs, ys, c=zs)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title("{} - {} datas".format(name, len(xs)))
    plt.colorbar()
plt.suptitle(z_key)
plt.savefig("tmp.png")
plt.clf()


fig = plt.figure(figsize=(8, 5))
for data_i, (data, name) in enumerate(zip(reversed(datas), reversed(names))):
    xs = data[x_key]
    ys = data[y_key]
    alpha = 0.2 if data_i < len(datas)-1 else 1.0
    label = 'aug' if data_i < len(datas)-1 else 'org'
    plt.scatter(xs, ys, label=label, alpha=0.5)
plt.legend(loc='best')
plt.xlabel(x_key)
plt.ylabel(y_key)
plt.title(z_key)
plt.savefig("tmp_2.png")
plt.clf()


exit()

mode = int(sys.argv[1])
if mode == 0:
    ## load data
    housing = pandas.read_csv(
        ## http://jse.amstat.org/v19n3/decock.pdf
        'https://raw.githubusercontent.com/nickkunz/smogn/master/data/housing.csv'
    )
    housing_smogn = smogn.smoter(
        data = housing,  ## pandas dataframe
        y = 'SalePrice',  ## string ('header name')
    )
    print(len(housing.columns))
    print(housing.shape)
    print(housing_smogn.shape)
else:
    ## load data
    csv_file = "dataset/csv/raw.csv"
    csv_data = pandas.read_csv(csv_file)
    csv_data = csv_data.drop(columns=['cell_id'])
    
    num_data = len(csv_data['cathode_loading_density'].to_list())
    csv_data_smogn = smogn.smoter(
        data = csv_data, 
        # y = 'cathode_loading_density', 
        # rel_coef = 0.05,
        y = 'specific_capacity_0.1C',
        k = 9,
        rel_thres = 0.5,          ## relevance threshold considered rare (pos real)
    )
    print(csv_data.shape)
    print(csv_data_smogn.shape)
exit()

## conduct smogn
smogn_data = smogn.smoter(
    data = csv_data, 
    y = "cathode_loading_density",
    k = 9,
    pert = 0.05,              ## perturbation / noise percentage (pos real)
    samp_method = "balance",  ## over / under sampling ("balance" or extreme")
    under_samp = True,        ## under sampling (bool)
    drop_na_col = True,       ## auto drop columns with nan's (bool)
    drop_na_row = True,       ## auto drop rows with nan's (bool)
    replace = False,          ## sampling replacement (bool)
    ## phi relevance function arguments / inputs
    rel_thres = 0.5,          ## relevance threshold considered rare (pos real)
    rel_method = "auto",      ## relevance method ("auto" or "manual")
    rel_xtrm_type = "both",   ## distribution focus ("high", "low", "both")
    rel_coef = 1.5,           ## coefficient for box plot (pos real)
    rel_ctrl_pts_rg = None    ## input for "manual" rel method  (2d array)
)
print(csv_data.shape)
print(smogn_data.shape)


