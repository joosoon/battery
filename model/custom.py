import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
        assert len(cfg["MODEL"]["ENCODER"]) == len(cfg["DATASET"]["INPUT"])
        assert len(cfg["MODEL"]["DECODER"]) == len(cfg["DATASET"]["OUTPUT"])

        input_name2data = {data["NAME"]: data for data in cfg["DATASET"]["INPUT"]}
        output_name2data = {data["NAME"]: data for data in cfg["DATASET"]["OUTPUT"]}
        # load encoders
        # (SIZE: input > hidden > ... > hidden > feature)
        name2out_size = {}
        self.encoders = {}
        for block in cfg["MODEL"]["ENCODER"]:
            # get module info
            name       = block.get("NAME", None)
            num_layer  = block.get("LAYER", None)
            size_h     = block.get("HIDDEN", None)
            size_f     = block.get("FEATURE", None)
            activation = block.get("ACTIVATION", None)
            # get size of input of module with DATASET-INPUT
            size_i = len(input_name2data[name]["ITEM"])
            # add module
            if block["TYPE"] == "MLP":
                self.encoders[name] = self._make_block_fc(num_layer, size_i, size_h, size_f, activation).to(device)
            elif block["TYPE"].endswith("LSTM"):
                config = block
                self.encoders[name] = self._make_block_lstm(config, device).to(device)
            # keep module info
            name2out_size[name] = size_f

        # add FUSION MODULE 
        # (encoder_outsize & add or concat > hidden > feature)
        self.fusions = {}
        self.fusion_name2info = {}
        if "FUSION" in self.cfg["MODEL"]:
            for block in cfg["MODEL"]["FUSION"]:
                # get module info
                name       = block["NAME"]
                num_layer  = block["LAYER"]
                size_h     = block["HIDDEN"]
                size_f     = block["FEATURE"]
                activation = block["ACTIVATION"]
                # get size of input of module with FUNCTION and ITEM
                func  = block["FUNCTION"]
                items = block["ITEM"]
                out_sizes = []
                for item in items:
                    out_size = name2out_size[item]
                    out_sizes.append(out_size)
                if func == 'add':
                    assert min(out_sizes) == max(out_sizes)
                    size_i = min(out_sizes)
                elif func == 'concat':
                    size_i = sum(out_sizes)
                # add module
                if block["TYPE"] == "MLP":
                    self.fusions[name] = self._make_block_fc(num_layer, size_i, size_h, size_f, activation).to(device)
                # keep module info
                self.fusion_name2info[name] = {"FUNCTION": func, "ITEM": items}
                name2out_size[name] = size_f
        
        # load decoders
        self.decoders = {}
        self.decoder_name2info = {}
        for block in cfg["MODEL"]["DECODER"]:
            # get module info
            name       = block["NAME"]
            num_layer  = block["LAYER"]
            size_h     = block["HIDDEN"]
            activation = block["ACTIVATION"]
            # get size of input of module with FUNCTION and ITEM
            func  = block["FUNCTION"]
            items = block["ITEM"]
            out_sizes = []
            for item in items:
                out_size = name2out_size[item]
                out_sizes.append(out_size)
            if func == 'add':
                assert min(out_sizes) == max(out_sizes)
                size_i = min(out_sizes)
            elif func == 'concat':
                size_i = sum(out_sizes)
            elif func == 'pass' and len(out_sizes)==1:
                size_i = out_sizes[0]
            # get size of otuput of module with DATASET-OUTPUT
            size_o = len(output_name2data[name]["ITEM"])
            # add module
            if block["TYPE"] == "MLP":
                self.decoders[name] = self._make_block_fc(num_layer, size_i, size_h, size_o, activation, no_acti_last=True).to(device)
            # keep module info
            self.decoder_name2info[name] = {"FUNCTION": func, "ITEM": items}
        # print("[ENCODER]:", self.encoders.keys())
        # print("[FUSION]:", self.fusions.keys())
        # print("[DECODER]:", self.decoders.keys())
        # print("---" * 20)

    def _make_block_lstm(self, config, device):
        if config["TYPE"] == "PlainLSTM":
            return PlainLSTM(config, device)
        elif config["TYPE"] == "AttnLSTM":
            return AttnLSTM(config, device)
        else:
            raise ValueError("Wrong TYPE of LSTM at CONFIG: {}".format(config["TYPE"]))

    def _make_block_fc(self, layers, i_size, h_size, o_size, 
                       activation="GELU", no_acti_last=False):
        block = []
        for i in range(layers):
            s = i_size if i == 0 else h_size
            e = o_size if i == layers-1 else h_size
            block.append(nn.Linear(s, e))
            if no_acti_last and i == layers-1: continue
            else:
                block.append(ACTIVATIONS[activation])
        return nn.Sequential(*block)        
    
    def get_params(self):
        params = []
        for layers in [self.encoders, self.fusions, self.decoders]:
            for k, v in layers.items():
                params += (list(layers[k].parameters()))
        return params
    
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
      
    def forward(self, input):
        # extract features using ENCODER modules
        features = {}
        for k, v in input.items():
            feature = self.encoders[k](v)
            features[k] = feature
            # print(k, feature.shape, v.shape)
        # print("---" * 20)
        
        # fusing features using FUSION modules
        for name, info in self.fusion_name2info.items():
            func  = info["FUNCTION"]
            items = info["ITEM"]
            if func=='add':
                feature = None
                for item in items:
                    if feature is None: 
                        feature = features[item]
                    else:
                        feature += features[item]
            elif func=='concat':
                feature = [features[item] for item in items]
                feature = torch.cat(feature, dim=1)
            out = self.fusions[name](feature)
            features[name] = out
        #     print(name, func, items, feature.shape, out.shape)
        # print("---" * 20)
            
        # get output using DECODER modules
        outs = {}
        for name, info in self.decoder_name2info.items():
            func  = info["FUNCTION"]
            items = info["ITEM"]
            if func=='add':
                feature = None
                for item in items:
                    if feature is None: 
                        feature = features[item]
                    else:
                        feature += features[item]
            elif func=='concat':
                feature = [features[item] for item in items]
                feature = torch.cat(feature, dim=1)
            elif func=='pass' and len(items)==1:
                feature = features[items[0]]
            out = self.decoders[name](feature)
            outs[name] = out
        #     print(name, func, items, feature.shape, out.shape)
        # print("---" * 20)
        return outs
            
            

class MLPOnly(nn.Module):
    def __init__(self, device):
        super(MLPOnly, self).__init__()
        self.encoder = nn.Sequential(
                       nn.Linear(6, 1024),
                       nn.GELU()).to(device)
        blocks = []
        for i in range(27):
            blocks.append(nn.Linear(1024, 1024))
            blocks.append(nn.GELU())
        self.fusion = nn.Sequential(*blocks).to(device)
        self.decoder = nn.Linear(1024, 7).to(device)
            
    def get_params(self):
        return self.parameters()
    
    def get_ckp(self):
        return self.state_dict()
    
    def load_ckp(self, state_dict):
        self.load_state_dict(state_dict)
        
    def forward(self, input_data):
        input_data = input_data['dp_all']
        feature = self.encoder(input_data)
        feature = self.fusion(feature)
        output = {'s_all': self.decoder(feature)}
        return output