from itertools import count
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from .resnet1d import ResNetFeature
from .lstm import PlainLSTM, AttnLSTM

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * \
               (x + 0.044715 * torch.pow(x, 3))))

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

ACTIVATIONS = {
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
}

class CUSTOM_FUSION(nn.Module):
    def __init__(self, cfg, device, hidden_size=30, feature_size=30):
        super(CUSTOM_FUSION, self).__init__()
        self.cfg = cfg
        assert len(cfg["MODEL"]["ENCODER"]["BLOCK"]) == len(cfg["DATASET"]["INPUT"])
        assert len(cfg["MODEL"]["DECODER"]["BLOCK"]) == len(cfg["DATASET"]["OUTPUT"])

        self.id2encoder_name, self.id2out_size = {}, {}
        # add ENCODER (SIZE: input > hidden > ... > hidden > feature)
        self.encoders = {}
        blocks = self.cfg["MODEL"]["ENCODER"]["BLOCK"]
        for data_idx, datas in enumerate(self.cfg["DATASET"]["INPUT"]):
            if datas["DATA"] == "META":
                key = "{}_{}".format(datas["DATA"], data_idx)
                size_i = len(datas["ITEM"])
            elif datas["DATA"] == "TIME":
                size_i = int(datas["LEN"])
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            elif datas["DATA"] == "CYCLE":
                size_i = 8 if datas["TYPE"] == "avg" else 40
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            # make block
            size_f = blocks[data_idx]["FEATURE"]
            activation = blocks[data_idx].get("ACTIVATION", None)
            if blocks[data_idx]["TYPE"] == "MLP":
                num_layer = blocks[data_idx]["LAYER"]
                size_h = blocks[data_idx]["HIDDEN"]
                self.encoders[key] = self._make_block_fc(num_layer, size_i, size_h, size_f, activation).to(device)
            elif blocks[data_idx]["TYPE"] == "CONV1D":
                settings = blocks[data_idx]["SETTING"]
                use_bn = blocks[data_idx].get("USE_BN", False)
                self.encoders[key] = self._make_block_conv1d(settings, activation).to(device)
            elif blocks[data_idx]["TYPE"] == "RESNET1D":
                config = blocks[data_idx]
                self.encoders[key] = self._make_block_resnet1d(config).to(device)
            elif blocks[data_idx]["TYPE"][-4:] == "LSTM":
                config = blocks[data_idx]
                self.encoders[key] = self._make_block_lstm(config, device).to(device)
            # TODO: ADDING MODEL TYPES
            else:
                raise ValueError("Wrong Block Type of Encoder {}".format(
                    blocks[data_idx]["TYPE"]))
            self.id2encoder_name[str(data_idx)] = key
            self.id2out_size[str(data_idx)] = size_f
        print("---" * 20)
        print("[ENCODER]:", self.encoders.keys())
        print("... ", self.id2encoder_name)
        self.last_feature_block = key
        # add FUSION MODULE (feature_size * num_input -> feature_size)
        # get orders
        self.fusions = {}
        if "FUSION" in self.cfg["MODEL"]:
            orders = self.cfg["MODEL"]["FUSION"]["ORDER"]
            self.id2block, self.id2items = self._split_orders(orders)
            self.block2id = {v:k for k, v in self.id2block.items()}
            blocks = self.cfg["MODEL"]["FUSION"]["BLOCK"]
            for id in reversed(list(self.id2block.keys())):
                block = self.id2block[id]
                items = self.id2items[id]
                # get size of input
                out_sizes = [self.id2out_size[item] for item in items]
                if block.split("_")[0] == "ADD":
                    assert min(out_sizes) == max(out_sizes), \
                        "Wrong Feature size of FUSION {} in config !".format(block)
                    size_i = min(out_sizes)
                elif block.split("_")[0] == "CONCAT":
                    size_i = sum(out_sizes)
                num_layer = blocks[block]["LAYER"]
                size_h = blocks[block]["HIDDEN"]
                size_f = blocks[block]["FEATURE"]
                activation = blocks[block]["ACTIVATION"]
                # add fusion layer
                key = block
                if blocks[block]["TYPE"] == "MLP":
                    self.fusions[key] = self._make_block_fc(num_layer, size_i, size_h, size_f, activation).to(device)
                # TODO: ADDING MODEL TYPES
                else: raise ValueError("Wrong Block Type of FUSION {}".format(blocks[block]["TYPE"]))
                self.id2out_size[block] = size_f
                self.id2out_size["FUSION"] = size_f
            self.last_feature_block = key
            print("---" * 20)
            print("[FUSION]:", self.fusions.keys())
            print("... FUSION ORDER:", orders)
            print("... ", self.id2items)
            print("... ", self.id2block)
            print("... ", self.block2id)
        # add DECODER (SIZE: input > hidden > ... > hidden > feature)
        self.decoders = {}
        blocks = self.cfg["MODEL"]["DECODER"]["BLOCK"]
        if "FUSION" in self.id2out_size:
            size_i = self.id2out_size["FUSION"]  
        else:
            size_i = self.id2out_size[list(self.id2out_size.keys())[0]]  
        for data_idx, datas in enumerate(self.cfg["DATASET"]["OUTPUT"]):
            if datas["DATA"] == "META":
                key = "{}_{}".format(datas["DATA"], data_idx)
                size_o = len(datas["ITEM"])
            elif datas["DATA"] == "TIME":
                size_o = int(datas["LEN"])
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            elif datas["DATA"] == "CYCLE":
                size_o = 8 if datas["TYPE"] == "avg" else 40
                key = "{}_{}_{}".format(datas["DATA"], datas["TYPE"], datas["ITEM"])
            num_layer = blocks[data_idx]["LAYER"]
            size_h = blocks[data_idx]["HIDDEN"]
            activation = blocks[data_idx]["ACTIVATION"]
            if blocks[data_idx]["TYPE"] == "MLP":
                self.decoders[key] = self._make_block_fc(num_layer, size_i, size_h, size_o, activation).to(device)
            # TODO: ADDING MODEL TYPES
            else:
                raise ValueError("Wrong Block Type of Encoder {}".format(
                    blocks[data_idx]["TYPE"]))
            self.id2out_size[str(data_idx)] = size_o
        print("---" * 20)
        print("[DECODER]:", self.decoders.keys())

    def _split_orders(self, orders):
        block_s, block_ids, block_id_last = 0, {}, 0
        item_s, item_ids = 0, []
        id2items, id2block = {}, {}
        for i, char in enumerate(orders):
            # find block
            if char == "(":
                block = orders[block_s:i]
                block_id_last += 1
                if block_id_last not in block_ids:
                    block_ids[block_id_last] = [0]
                block_ids[block_id_last].append(max(block_ids[block_id_last])+1)
                block_id = "{}.{}".format(block_id_last, max(block_ids[block_id_last]))
                block_id = float(block_id)
                block_s = i+1
                item_ids.append(block_id)
                id2block[block_id] = block
            elif char == ",":
                block_s = i+1
            elif char == ")":
                block_s = i+1
                block_id_last -= 1

            # find item
            if char == "(":
                item = orders[item_s:i]
                item_s = i+1
                if len(item_ids) < 2: continue
                item_id = item_ids[-2]
                if item_id not in id2items: id2items[item_id] = []
                id2items[item_id].append(item)
            elif char == ",":
                if item_s==i: 
                    item_s = i+1
                    continue
                item = orders[item_s:i]
                item_s = i+1
                item_id = item_ids[-1]
                if item_id not in id2items: id2items[item_id] = []
                id2items[item_id].append(item)
            elif char == ")":
                if item_s==i: continue
                item = orders[item_s:i]
                item_s = i+1
                item_id = item_ids[-1]
                item_ids = item_ids[:-1]
                if item_id not in id2items: id2items[item_id] = []
                id2items[item_id].append(item)
        return id2block, id2items

    def _make_block_lstm(self, config, device):
        if config["TYPE"] == "PlainLSTM":
            return PlainLSTM(config, device)
        elif config["TYPE"] == "AttnLSTM":
            return AttnLSTM(config, device)
        else:
            raise ValueError("Wrong TYPE of LSTM at CONFIG: {}".format(config["TYPE"]))

    def _make_block_resnet1d(self, config):
        return ResNetFeature(config)

    def _make_block_conv1d(self, settings, activation="GELU", use_bn=False):
        return CONV1D(settings, activation, use_bn)
            
    def _make_block_fc(self, layers, i_size, h_size, o_size, activation="GELU"):
        block = []
        for i in range(layers):
            s = i_size if i == 0 else h_size
            e = o_size if i == layers-1 else h_size
            block.append(nn.Linear(s, e))
            if i < layers-1:
                block.append(ACTIVATIONS[activation])
        return nn.Sequential(*block)

    def get_params(self):
        params = []
        for layers in [self.encoders, self.fusions, self.decoders]:
            for k, v in layers.items():
                params += (list(layers[k].parameters()))
        return params

    def forward(self, input):
        # print(input.keys())
        # print(self.encoders.keys())
        # print(self.fusions.keys())
        # print(self.decoders.keys())
        # print("---" * 20)

        # ENCODER
        # print("... ENCODER")
        features = {}
        for k, v in input.items():
            feature = self.encoders[k](v)
            features[k] = feature
            # print(feature.shape, k)
        # FUSION
        # print("---" * 20)
        # print("... FUSION")
        for k in self.fusions.keys():
            fusion_id = self.block2id[k]
            items = []
            for item in self.id2items[fusion_id]:
                if item in self.id2encoder_name:
                    items.append(self.id2encoder_name[item])
                else:
                    items.append(item)
            if k.split("_")[0] == "ADD":
                feature = None
                for i, item in enumerate(items):
                    if i == 0: feature = features[item]
                    else: feature += features[item]
                feature /= len(items)
            elif k.split("_")[0] == "CONCAT":
                feature = [features[item] for item in items]
                feature = torch.cat(feature, dim=1)
            feature = self.fusions[k](feature)
            features[k] = feature
            # print(feature.shape, k)
        # DECODER (feature -> decoder)
        # print(features.keys())
        # print(self.last_feature_block)
        # print("---" * 20)
        feature = features[self.last_feature_block]
        outs = {}
        for k in self.decoders.keys():
            out = self.decoders[k](feature)
            outs[k] = out
            # print(out.shape, k)
        return outs      
    
    def get_ckp(self):
        params = {}
        # get params of encoders
        params["encoder"] = {}
        for k, v in self.encoders.items():
            params["encoder"][k] = v.state_dict()
        # get params of fusions
        params["fusion"] = {}
        for k, v in self.fusions.items():
            params["fusion"][k] = v.state_dict()
        # get params of decoders
        params["decoder"] = {}
        for k, v in self.decoders.items():
            params["decoder"][k] = v.state_dict()
        return params

    def load_ckp(self, state_dict):
        # load encoders
        for k, v in self.encoders.items():
            v.load_state_dict(state_dict["encoder"][k])
        # load fusions
        for k, v in self.fusions.items():
            v.load_state_dict(state_dict["fusion"][k])
        # load decoders
        for k, v in self.decoders.items():
            v.load_state_dict(state_dict["decoder"][k])
      
    def set_mode(self, mode="train"):
        assert mode in ["train", "test"]
        for layers in [self.encoders, self.fusions, self.decoders]:
            for k, v in layers.items():
                if mode=="train": v.train()
                else: v.eval()
                
class CONV1D(nn.Module):
    def __init__(self, settings, activation="GELU", use_bn=True,
                 device="cuda", no_activation_last=False):
        super(CONV1D, self).__init__()
        self.use_bn = use_bn
        self.activation = ACTIVATIONS[activation]
        self.no_activation_last = no_activation_last
        self.num_layer = len(settings)
        self.bn_layers = {}
        self.conv_layers = {}
        for i, setting in enumerate(settings):
            in_channels, out_channels, kernel_size, stride, padding = setting
            self.conv_layers[i] = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding).to(device)
            if self.use_bn:
                self.bn_layers[i] = nn.BatchNorm1d(out_channels).to(device)
                
        # count the number of parameters        
        param_sum = 0
        for i, layer in self.conv_layers.items():
            for p in layer.parameters():
                if p.requires_grad:
                    param_sum += p.numel()
        print(param_sum)

    def forward(self, x):
        if len(x.shape) == 2: # shape must be (batch, channel, data)
            x = x.unsqueeze(1)
        for i, conv_layer in self.conv_layers.items():
            x = conv_layer(x)
            if self.use_bn: x = self.bn_layers[i](x)
            if (i+1)==self.num_layer and self.no_activation_last: continue
            else: x = self.activation(x)
        x = x.squeeze(1)
        return x
