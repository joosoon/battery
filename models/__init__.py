from .custom import *
from .losses import *
import torch.nn as nn

def load_model(cfg, device):
    if cfg["MODEL"]["NAME"] == "CUSTOM_FUSION":
        model = CUSTOM_FUSION(cfg, device)
    else:
        raise ValueError("Wrong MODEL Name from config [{}]".format(cfg["MODEL"]["NAME"]))
    return model

def load_criterions(cfg, device):
    critreion = CriterionLoss(cfg)
    if cfg["LOSS"] == "MSE": 
        loss = nn.MSELoss()
    elif cfg["LOSS"] == "CE":
        loss = nn.CrossEntropyLoss()
    elif cfg["LOSS"] == "BCE":
        loss = nn.BCELoss()
    else:
        raise ValueError("Wrong Loss Name from config [{}]".format(cfg["LOSS"]))
    return loss
