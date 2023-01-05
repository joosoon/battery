import os
from platform import dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import time
import pickle


def draw_save_graph(data_x, data_y, data_label, save_name):
    class_ids = np.unique(data_label)
    palette = sns.color_palette("Paired", len(class_ids))
    data = pd.DataFrame(index = range(len(data_x)))
    data['x']=data_x
    data['y']=data_y
    data['label']=data_label
    plt.figure(figsize=(8,8))
    sns.scatterplot(x='x', y='y', hue='label', palette=palette, legend='full', data=data)
    plt.savefig(save_name)
    plt.clf()

def get_tsne(datas):
    t_1 = time.time()
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(datas)
    tsne_x, tsne_y = np.split(tsne_results, 2, axis=1)
    print("... T-SNE analysis: {:.2f}s".format(time.time()-t_1))
    return tsne_x, tsne_y


def draw_numpy_tsne(data_root, data_len=None, save_name=None, target_labels=None):
    data_list = sorted(os.listdir(data_root))
    datas, labels = [], []
    for data_name in data_list:
        data_np = os.path.join(data_root, data_name)
        data_np = np.load(data_np)
        if data_len: 
            if len(data_np) >= data_len:
                data_np = data_np[:data_len]
            else: continue
        data_label = data_name[:3]
        if (not target_labels) or (target_labels and data_label in target_labels):
            datas.append(data_np)
            labels.append(data_label)
    datas = np.vstack(datas)
    labels = np.array(labels)
    class_ids = np.unique(labels)

    print("+++ start visualing {} data, {} categories".format(datas.shape, len(class_ids)))
    if not save_name:
        save_name = "tsne_{}.png".format(data_root.replace("/", "_").replace(".", ""))
    tsne_x, tsne_y = get_tsne(datas)
    draw_save_graph(tsne_x, tsne_y, labels, save_name)


def draw_pickle_meta_tsne(meta_file, data_items=None, save_name=None, target_labels=None):
    with open(meta_file, 'rb') as f:
        meta_data = pickle.load(f)
    cell_ids = [cell for cell in meta_data.keys()]
    if not data_items:
        data_items = [item_name for item_name in meta_data[cell_ids[0]]]

    datas, labels = [], []
    for cell_id in cell_ids:
        data = [meta_data[cell_id][data_item] for data_item in data_items]
        datas.append(data)
        labels.append(cell_id[:3])
    datas = np.array(datas)
    labels = np.array(labels)
    class_ids = np.unique(labels)

    print("+++ start visualing {} data, {} categories".format(datas.shape, len(class_ids)))
    if not save_name:
        save_name = "tsne_{}.png".format(meta_file.replace("/", "_").replace(".", ""))
    tsne_x, tsne_y = get_tsne(datas)
    draw_save_graph(tsne_x, tsne_y, labels, save_name)

if __name__ == '__main__':
    # fix numpy seed
    np.random.seed(77)

    targets = ["1.1","2.2","3.3"]

    data_root = "./data/numpy/time/norm_split/전압_방전_cycle0"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, 1000, save_name, targets)
    data_root = "./data/numpy/time/norm_split/전압_방전_cycle1"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, 1000, save_name, targets)    
    data_root = "./data/numpy/time/norm_split/전압_방전_cycle2"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, 1000, save_name, targets)
    data_root = "./data/numpy/time/norm_split/전압_방전_cycle3"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, 1000, save_name, targets)



    exit()


    ###################
    # draw NUMPY DATA #
    ###################
    # TIME DATA: 전압-방전-cycle2
    data_root = "./data/numpy/time/norm_split/전압_방전_cycle2"
    draw_numpy_tsne(data_root, 1265)

    data_root = "./data/numpy/cycle/avg_norm/방전 용량_s(Ah-g)"
    draw_numpy_tsne(data_root)

    data_root = "./data/numpy/cycle/norm/방전 용량_s(Ah-g)"
    draw_numpy_tsne(data_root, 39)

    data_root = "./data/numpy/cycle/norm/충전 용량_s(Ah-g)"
    draw_numpy_tsne(data_root, 39)


    targets = ["1.1","1.2","1.3"]
    data_root = "./data/numpy/time/norm_split/전압_방전_cycle2"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, 1265, save_name, targets)

    data_root = "./data/numpy/cycle/avg_norm/방전 용량_s(Ah-g)"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, None, save_name, targets)

    data_root = "./data/numpy/cycle/norm/방전 용량_s(Ah-g)"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, 39, save_name, targets)

    data_root = "./data/numpy/cycle/norm/충전 용량_s(Ah-g)"
    save_name = "tsne_" + data_root.replace(".", "").replace("/", "_")
    save_name += "_Target_" + "_".join(targets) + ".png"
    draw_numpy_tsne(data_root, 39, save_name, targets)
    exit()

    #########################
    # draw MET(PICKLE) DATA #
    #########################
    meta_file = './data/pickle/cell_meta_norm.pickle'
    draw_pickle_meta_tsne(meta_file)

    meta_file = './data/pickle/cell_meta_norm.pickle'
    data_items = [
        'anode_전극 두께 (㎛)', 'anode_AM thickness (㎛)', 'anode_Weight (mg)', 
        'anode_Electrode weight (w/o foil,g)', 'anode_Loading mass of AM (mg)', 
        'anode_Loading (mg/cm2)', 'anode_Porosity', 'anode_Loading density (mAh/cm2)', 
        'anode_Theoretical capacity (mAh)']
    save_name = "tsne__data_pickle_cell_meta_normpickle_anode.png"
    draw_pickle_meta_tsne(meta_file, data_items, save_name)

    meta_file = './data/pickle/cell_meta_norm.pickle'
    data_items = [
        'cathod_전극 두께 (㎛)', 'cathod_AM thickness (㎛)', 'cathod_Weight (mg)', 
        'cathod_Electrode weight (w/o foil,g)', 'cathod_Loading mass of AM (mg)', 
        'cathod_Loading (mg/cm2)', 'cathod_Porosity', 'cathod_Loading density (mAh/cm2)', 
        'cathod_Theoretical capacity (mAh)']
    save_name = "tsne__data_pickle_cell_meta_normpickle_cathod.png"
    draw_pickle_meta_tsne(meta_file, data_items, save_name)
