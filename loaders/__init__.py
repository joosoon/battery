from .batterycell import *

def load_dataset(cfg, k_fold, mode="train"):
    assert (k_fold is None) or (k_fold < 9 and k_fold >= 0)
    if cfg["DATASET"]["NAME"] == "full_cell":
        datast = BatteryCellData(cfg, k_fold, mode)
        coll_fn = collate_batterycell
    else:
        raise ValueError("Wrong dataset name of config [{}]".format(cfg["DATASET"]["NAME"]))
    return datast, coll_fn
