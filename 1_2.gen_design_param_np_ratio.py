import os 
import pandas as pd
import numpy as np
import argparse
import math

        
class SynDesignParam():
    def __init__(self, samples):
        self.param = {}
        for target, val in samples.items():
            self.param[target] = np.array(val)
        self.num_data = len(val)
        # calculate cathode param
        self.param['cathode_Electrode weight (w/o foil,g)'] = self.param['cathode_loading_density'] / 1000 * 0.25 * math.pi
        self.param['cathode_Weight (mg)'] = self.param['cathode_Electrode weight (w/o foil,g)'] * 1000 + 6.4
        self.param['cathode_Loading mass of AM (mg)'] = self.param['cathode_Weight (mg)']-6.4
        self.param['cathode_Loading mass of AM (mg)'] *= 0.93
        self.param['cathode_Loading mass of AM (mg)'] = np.round(self.param['cathode_Loading mass of AM (mg)'], 2)
        self.param['cathode_Loading density (mAh/cm2)'] = self.param['cathode_Loading mass of AM (mg)'] * 210 / 0.25 / math.pi / 1000
        self.param['cathode_Theoretical capacity (mAh)'] = self.param['cathode_Loading mass of AM (mg)']*210
        self.param['cathode_Theoretical capacity (mAh)'] = np.round(self.param['cathode_Theoretical capacity (mAh)'], -1)
        self.param['cathode_AM_thickness']  = self.param['cathode_Electrode weight (w/o foil,g)'].copy()
        self.param['cathode_AM_thickness'] /= 1 - self.param['cathode_porosity'] / 100
        self.param['cathode_AM_thickness'] *= 10000 / math.pi / 0.25
        self.param['cathode_AM_thickness'] *= (0.93/4.7) + (0.04/1.9) + (0.03/1.78)
        self.param['cathode_전극 두께 (㎛)'] = self.param['cathode_AM_thickness'] + 30
        # calculate anode param
        self.param['anode_Theoretical capacity (mAh)'] = self.param['cathode_Theoretical capacity (mAh)'].copy()
        self.param['anode_Theoretical capacity (mAh)'] *= self.param['np_ratio'] * 196 / 100
        self.param['anode_Theoretical capacity (mAh)'] = np.round(self.param['anode_Theoretical capacity (mAh)'], -1)
        self.param['anode_Loading mass of AM (mg)'] = self.param['anode_Theoretical capacity (mAh)'] / 360
        self.param['anode_Loading mass of AM (mg)'] = np.round(self.param['anode_Loading mass of AM (mg)'], 2)
        self.param['anode_Weight (mg)'] = self.param['anode_Loading mass of AM (mg)'] / 0.96
        self.param['anode_Weight (mg)'] += 45
        self.param['anode_Weight (mg)'] = np.round(self.param['anode_Weight (mg)'], 1)
        self.param['anode_Electrode weight (w/o foil,g)'] = self.param['anode_Weight (mg)'] - 45
        self.param['anode_Electrode weight (w/o foil,g)'] /= 1000
        self.param['anode_loading_density'] = self.param['anode_Electrode weight (w/o foil,g)'] * 1000 / 0.49 / math.pi
        self.param['anode_AM_thickness']  = self.param['anode_Electrode weight (w/o foil,g)'].copy()
        self.param['anode_AM_thickness'] /= 1 - self.param['anode_porosity']
        self.param['anode_AM_thickness'] *= 10000 / math.pi / 0.49
        self.param['anode_AM_thickness'] *= (0.96/2.26) + (0.01/1.95) + (0.015/1.254)
        
        
if __name__ == '__main__':
    save_file = 'dataset/csv/syn_np_ratio_search.csv'    
    # uniform sampling in MIN-MAX range
    target2sample = {}
    # cathode_loading_density
    key = 'cathode_loading_density'
    min_val, max_val, num_sample = 5, 20, 50
    target2sample[key] = np.linspace(min_val, max_val, num_sample)
    # cathode_porosity
    key = 'cathode_porosity'
    min_val, max_val, num_sample = 15, 40, 50
    target2sample[key] = np.linspace(min_val, max_val, num_sample)
    # anode_porosity
    key = 'anode_porosity'
    min_val, max_val, num_sample = 0.2, 0.45, 50
    target2sample[key] = np.linspace(min_val, max_val, num_sample)
    # np_ratio
    key = 'np_ratio'
    # target2sample[key] = np.array([1.1, 1.2])
    min_val, max_val, num_sample = 1.0, 1.3, 4
    target2sample[key] = np.linspace(min_val, max_val, num_sample)

    # generate combinations of sample
    samples_all = []
    for target, samples in target2sample.items():
        samples_all.append(samples)
    combs = np.array(np.meshgrid(*samples_all)).T.reshape(-1, len(samples_all))
    print("... generate {} combinations".format(len(combs)))

    # set combinations
    param_sample = {}
    for i, target in enumerate(target2sample.keys()):
        param_sample[target] = combs[:, i]

    # generate synthetic design param
    param_syn = SynDesignParam(param_sample)
    
    # save as CSV
    save_keys = [
                'cathode_loading_density',
                'cathode_porosity',
                'cathode_AM_thickness',
                'anode_loading_density',
                'anode_porosity',
                'anode_AM_thickness'
                 ]

    save_dict = {}
    save_dict['cell_id'] = list(range(param_syn.num_data))
    for k in save_keys:
        v = param_syn.param[k]
        save_dict[k] = v.copy()
    df = pd.DataFrame(save_dict).set_index('cell_id')
    df.to_csv(save_file)
    print("... saved at {}".format(save_file))
 
 
 
 
 
 

    # save_file = 'dataset/csv/syn_np_ratio_vis_fix_anode_mini.csv'
    # save_file = 'dataset/csv/syn_np_ratio_vis_fix_anode.csv'
    # # uniform sampling in MIN-MAX range
    # target2sample = {}
    # # cathode_loading_density
    # key = 'cathode_loading_density'
    # min_val, max_val, num_sample = 10, 20, 500
    # target2sample[key] = np.linspace(min_val, max_val, num_sample)
    # # cathode_porosity
    # key = 'cathode_porosity'
    # min_val, max_val, num_sample = 15, 40, 250
    # target2sample[key] = np.linspace(min_val, max_val, num_sample)
    # # anode_porosity
    # key = 'anode_porosity'
    # min_val, max_val, num_sample = 0.325, 0.325, 1
    # target2sample[key] = np.linspace(min_val, max_val, num_sample)
    # # np_ratio
    # key = 'np_ratio'
    # # target2sample[key] = np.array([1.1, 1.2])
    # min_val, max_val, num_sample = 1.24, 1.24, 1
    # target2sample[key] = np.linspace(min_val, max_val, num_sample)



    # #########################
    # # check synthetic param #
    # #########################
    # param_sample = {}
    # param_sample['cathode_loading_density'] = [10.4406,10.1859,10.5679,10.5679,10.4406,10.4406]
    # param_sample['cathode_porosity']        = [38.7646,31.9653,32.6572,31.3585,36.3911,30.0663]
    # param_sample['anode_porosity']          = [0.4071,0.4038,0.3983,0.4108,0.3705,0.4207]
    # param_sample['np_ratio']                = [1.1,1.1,1.1,1.1,1.1,1.1]
    # param_syn = SynDesignParam(param_sample)
    # key = 'cathode_AM_thickness'
    # datas = param_syn.param[key]
    # for data in datas:
    #     print(data)
    # exit()


    # # np_ratio_vis
    # key = 'cathode_porosity'
    # min_val, max_val, num_sample = 0.25, 0.35, 200
    # target2sample[key] = np.linspace(min_val, max_val, num_sample)
    # target2sample['anode_porosity'] = [0.3]
    # target2sample['np_ratio'] = [1.15]

    # # draw histogram
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(20, 10))
    # keys = ['loading_density', 'porosity', 'porosity']
    # for key_i, key in enumerate(keys):
    #     plt.subplot(2, 3, key_i+1)  
    #     plt.hist(save_dict["cathode_{}".format(key)], bins=30)
    #     plt.title("cathode_{}".format(key))
    #     plt.subplot(2, 3, key_i+4)  
    #     plt.hist(save_dict["anode_{}".format(key)], bins=30)
    #     plt.title("anode_{}".format(key))

    # save_name = "tmp_hist.png"
    # plt.savefig(save_name, dpi=500)
    # plt.clf()


    
    
    