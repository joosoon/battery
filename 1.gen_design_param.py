import os 
import pandas as pd
import numpy as np
import argparse

class SynDesignParam():
    def __init__(self, targets):
        self.param = {}
        for target, val in targets.items():
            self.param[target] = val
        self.num_data = len(val)

        # set porosity and loading_density
        check_target2val = {
                'porosity': 0.0135,
                'loading_density': 0.65642}
        for target, val in check_target2val.items():
            a = 'anode_{}'.format(target)
            c = 'cathode_{}'.format(target)
            if a not in targets and c in targets:
                self.param[a] = self.param[c] * val
            elif a in targets and c not in targets:
                self.param[c] = self.param[a] / val
        
        # calculate AM_thickness with Electro Weight
        if 'cathode_AM_thickness' not in self.param:
            # get Electro Weight
            c_EW = self.param['cathode_loading_density'] / 1000 * 0.25 * np.pi
            self.param['cathode_AM_thickness']  = c_EW
            self.param['cathode_AM_thickness'] /= 1 - self.param['cathode_porosity']/100
            self.param['cathode_AM_thickness'] *= 10000 / np.pi / 0.25
            self.param['cathode_AM_thickness'] *= (0.93/4.7) + (0.04/1.9) + (0.03/1.78)
            
        if 'anode_AM_thickness' not in self.param:
            # get Electro Weight
            a_EW = self.param['anode_loading_density'] / 1000 * 0.49 * np.pi
            self.param['anode_AM_thickness']  = a_EW
            self.param['anode_AM_thickness'] /= 1 - self.param['anode_porosity']
            self.param['anode_AM_thickness'] *= 10000 / np.pi / 0.49
            self.param['anode_AM_thickness'] *= (0.96/2.26) + (0.01/1.95) + (0.015/1.254)

        
if __name__ == '__main__':
    # the number of sampling
    num_sample = 30
    # offset of MIN, MAX val
    offset = 0.1       
    # target of synthesis
    targets = [
        'cathode_loading_density',
        'cathode_porosity',
        'anode_loading_density',
        'anode_porosity'
    ]

    save_file = 'dataset/csv/syn_t{}_s{}_o{}.csv'.format(len(targets), num_sample, offset)

    # load real data
    csv_file  = 'dataset/csv/raw.csv'
    csv_data = pd.read_csv(csv_file)
    target2data = {target: csv_data[target].to_numpy() for target in targets}

    # uniform sampling in MIN-MAX range
    print("... sampling {} datas in {} targets".format(num_sample, len(targets)))
    target2sample = {}
    for target, data in target2data.items():
        min_val = data.min() 
        max_val = data.max() 
        samples = np.linspace(min_val*(1-offset), max_val*(1+offset), num_sample)
        target2sample[target] = samples
    
    # generate combinations of sample
    samples_all = []
    for target, samples in target2sample.items():
        samples_all.append(samples)
    combs = np.array(np.meshgrid(*samples_all)).T.reshape(-1, len(samples_all))
    print("... gnerate {} combinations".format(len(combs)))
    
    # set combinations
    param_sample = {}
    for i, target in enumerate(target2sample.keys()):
        param_sample[target] = combs[:, i]
        
    # generate last parameters
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

