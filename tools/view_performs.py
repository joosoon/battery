import torch
import os


ckp_root = "./results"

model_types = [
    '211024_meta_reg', '211024_meta_reg_cls', 
    '211024_time_reg', '211024_time_reg_cls', 
    '211024_meta_time_reg', '211024_meta_time_reg_cls']
l_or_ms = ["loss", "metric"]
kfold = None

for model_type in model_types:
    for l_or_m in l_or_ms:
        ckp_file = os.path.join(
                    ckp_root, model_type, 
                    "{}_kfold_{}_best_{}.pth".format(model_type, kfold, l_or_m))
        ckp = torch.load(ckp_file)
        print("| EP:{} | LOSS:{:.3f} | METRIC: {:.3f} | {} | {}".format(
              ckp["epoch"], ckp["loss"], ckp["metric"], model_type, l_or_m))