import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from tqdm import tqdm
import numpy as np
import yaml

import torch
from loader.batterycell import collate_batterycell, BatteryCellData
from model.custom import CUSTOM_FUSION, MLPOnly
from model.loss import CriterionLoss

class ParserCRATE():
    def __init__(self, cfg):
        self.crates = ['0.1C','0.2C','0.5C','1C','2C','3C','5C']
        self.cfg = cfg
        self.output_idx2crate = {}
        for datas in self.cfg["DATASET"]["OUTPUT"]:
            name  = datas['NAME']
            if name not in self.output_idx2crate:
                self.output_idx2crate[name] = {}
            items = datas['ITEM']
            for idx, item in enumerate(items):
                self.output_idx2crate[name][idx] = item

    def parsing(self, data_dict):
        parse_dict = {}
        for k, v in data_dict.items():
            for data_idx in range(v.shape[1]):
                data_crate = self.output_idx2crate[k][data_idx]
                data_type = "_".join(data_crate.split("_")[:-1])
                crate = data_crate.split("_")[-1]
                if data_type not in parse_dict:
                    parse_dict[data_type] = {}
                parse_dict[data_type][crate] = v[:, data_idx]
        return parse_dict
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battery Performace')
    parser.add_argument('--cfg', type=str, default="config/base.yaml")
    # param for dataset 
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dataset_mode', type=str, default='test')
    # param for model
    parser.add_argument('--ckp_mode', type=str, default='latest', help="name/of/trained/weight")
    parser.add_argument('--ckp_file', type=str, default=None, help="path/to/trained/weight")
    # param for foward
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', default='cuda', choices=["cuda", "cpu"])
    # parma for visualization
    parser.add_argument('--vis_r2', action='store_true')
    parser.add_argument('--save_np', action='store_true')
    parser.add_argument('--prt_inference', action='store_true')
    args = parser.parse_args()

    # set device (GPU and CPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else: device = torch.device("cpu")

    # load config file
    if not args.cfg.endswith('.yaml'): args.cfg += '.yaml'
    assert os.path.isfile(args.cfg), "no config of {}".format(args.cfg)
    with open(args.cfg, encoding='UTF8') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        args.cfg = args.cfg[:-5]

    # set seed for reproduciblity
    seed = cfg["DATASET"].get("SEED", args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # load dataset
    coll_fn = collate_batterycell
    test_dataset = BatteryCellData(cfg, args.dataset_mode, seed)
    test_loader  = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=1, 
                        num_workers=0, shuffle=False, 
                        collate_fn=coll_fn)

    # load model
    model = CUSTOM_FUSION(cfg, device)
    # model = MLPOnly(device)
    if args.ckp_file is None:
        args.ckp_file = os.path.join(cfg["SAVE_ROOT"], "{}.pth".format(args.ckp_mode))
    ckp = torch.load(args.ckp_file, map_location=torch.device('cpu'))
    model.load_ckp(ckp['net'])
    
    # load criterion
    criterion = CriterionLoss(cfg)

    prs_all = {}
    gts_all = {}
    for idx, (data_input, data_gt) in enumerate(test_loader):
        # forward
        data_input = {k:v.to(device) for k, v in data_input.items()}
        data_gt = {k:v.to(device) for k, v in data_gt.items()}
        with torch.no_grad():
            outputs = model(data_input)
        # keep output
        for k in outputs.keys():
            output = outputs[k].clone().detach().cpu()
            if k not in prs_all:
                prs_all[k] = output
            else:
                prs_all[k] = torch.cat((prs_all[k], output))
        # keep gt
        for k in data_gt.keys():
            gt = data_gt[k].clone().detach().cpu()
            if k not in gts_all:
                gts_all[k] = gt
            else:
                gts_all[k] = torch.cat((gts_all[k], gt))
    
    # print("Epoch {} | Mode {}".format(ckp['epoch'], args.ckp_mode))
    # print("---" * 20)
    # compute METRIC
    prs_all_denorm = test_dataset.transform_output.denorm(prs_all)
    gts_all_denorm = test_dataset.transform_output.denorm(gts_all)
    for k in prs_all.keys():
        gts = gts_all[k]
        prs = prs_all[k]
        prs_dn = prs_all_denorm[k]
        gts_dn = gts_all_denorm[k]
        
        mae = criterion.criterion["MAE"](prs, gts).numpy()
        mse = criterion.criterion["MSE"](prs, gts).numpy()
        r2  = criterion.criterion["R2" ](prs, gts)
        
        mae_dn = criterion.criterion["MAE"](prs_dn, gts_dn).numpy()
        mse_dn = criterion.criterion["MSE"](prs_dn, gts_dn).numpy()
        r2_dn  = criterion.criterion["R2" ](prs_dn, gts_dn)
        
        # # print results
        # print("{},{},{},{}".format(k, mae_dn, mse_dn, r2_dn))
    
    # visualization of R2
    if args.vis_r2:
        import matplotlib.pyplot as plt    
        save_dir = os.path.join(cfg["SAVE_ROOT"], 'inference', 'R2')
        os.makedirs(save_dir, exist_ok=True)
        
        for k in prs_all.keys():
            prs_dn = prs_all_denorm[k]
            gts_dn = gts_all_denorm[k]
            r2_dn  = criterion.criterion["R2" ](prs_dn, gts_dn)

            save_name = os.path.join(save_dir, "{}_{}.png".format(k, args.ckp_mode))    
            plt.scatter(prs_dn, gts_dn)
            plt.title("{} | R2: {:.4f}".format(k, r2_dn))
            plt.xlabel("Prediction")
            plt.ylabel("Ground Truth")
            plt.savefig(save_name, dpi=500)
            plt.clf()
    
    
    # parsing into C-RATE
    crate_parser = ParserCRATE(cfg)
    prs_parsing = crate_parser.parsing(prs_all_denorm)
    gts_parsing = crate_parser.parsing(gts_all_denorm)



        
    # print("+++" * 20)
    for data_type in prs_parsing.keys():
        prs = prs_parsing[data_type]
        gts = gts_parsing[data_type]
        prs_crate_all = []
        gts_crate_all = []
        # print("{},MAE,MSE,R2".format(data_type))
        for crate in crate_parser.crates:
            if crate not in prs: continue
            pr = prs[crate]
            gt = gts[crate]
            prs_crate_all.append(pr)
            gts_crate_all.append(gt)
            
            mae = criterion.criterion["MAE"](pr, gt).numpy()
            mse = criterion.criterion["MSE"](pr, gt).numpy()
            r2  = criterion.criterion["R2" ](pr, gt)
            # print("{},{},{},{}".format(crate,mae,mse,r2))

            # print('---' * 20)
            # print("0 >>>", data_type, crate)
            # print(np.array(test_dataset.cell_ids)[[0,1,3,4]])
            # print(gt[[0,1,3,4]])
            # print('---' * 20)

            if args.vis_r2:
                import matplotlib.pyplot as plt    
                save_dir = os.path.join(cfg["SAVE_ROOT"], 'inference', 'R2')
                os.makedirs(save_dir, exist_ok=True)
                save_name = os.path.join(save_dir, "{}_{}_{}.png".format(data_type, args.ckp_mode, crate))    
                plt.scatter(pr, gt)
                plt.title("{} | {} | R2: {:.4f}".format(data_type, crate, r2))
                plt.xlabel("Prediction")
                plt.ylabel("Ground Truth")
                plt.savefig(save_name, dpi=500)
                plt.clf()

            if args.save_np:
                save_dir = os.path.join(cfg["SAVE_ROOT"], 'inference', 'numpy', data_type)
                os.makedirs(save_dir, exist_ok=True)
                save_name_gt = os.path.join(save_dir, "{}_{}_gt.npy".format(args.dataset_mode, crate))
                save_name_pr = os.path.join(save_dir, "{}_{}_pr.npy".format(args.dataset_mode, crate))
                np.save(save_name_gt, gt.cpu().numpy())
                np.save(save_name_pr, pr.cpu().numpy())

        if args.prt_inference:
            print("GT", "---" * 20)
            cell_ids = np.array(test_dataset.cell_ids)
            print(str(list(cell_ids)).replace("'","").replace("[","").replace("]",""))
            for crate in crate_parser.crates:
                if crate not in prs: continue
                gt = gts[crate]
                print(str(list(gt.cpu().numpy())).replace("'", "").replace("[","").replace("]",""))
            print("Pred", "---" * 20)
            cell_ids = np.array(test_dataset.cell_ids)
            print(str(list(cell_ids)).replace("'","").replace("[","").replace("]",""))
            for crate in crate_parser.crates:
                if crate not in prs: continue
                pr = prs[crate]
                print(str(list(pr.cpu().numpy())).replace("'", "").replace("[","").replace("]",""))
            exit()            

        prs_crate_all = torch.cat(prs_crate_all)
        gts_crate_all = torch.cat(gts_crate_all)
        mae = criterion.criterion["MAE"](prs_crate_all, gts_crate_all).numpy()
        mse = criterion.criterion["MSE"](prs_crate_all, gts_crate_all).numpy()
        r2  = criterion.criterion["R2" ](prs_crate_all, gts_crate_all)
        # print("ALL,{},{},{}".format(mae,mse,r2))

        if args.vis_r2:
            import matplotlib.pyplot as plt    
            save_dir = os.path.join(cfg["SAVE_ROOT"], 'inference', 'R2')
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, "{}_{}_ALL.png".format(data_type, args.ckp_mode))    
            plt.scatter(prs_crate_all, gts_crate_all)
            plt.title("{} | ALL | R2: {:.4f}".format(data_type, r2))
            plt.xlabel("Prediction")
            plt.ylabel("Ground Truth")
            plt.savefig(save_name, dpi=500)
            plt.clf()        

        if args.save_np:
            save_dir = os.path.join(cfg["SAVE_ROOT"], 'inference', 'numpy', data_type)
            os.makedirs(save_dir, exist_ok=True)
            save_name_gt = os.path.join(save_dir, "ALL_gt.npy")
            save_name_pr = os.path.join(save_dir, "ALL_pr.npy")
            np.save(save_name_gt, prs_crate_all.cpu().numpy())
            np.save(save_name_pr, gts_crate_all.cpu().numpy())

            cell_ids = np.array(test_dataset.cell_ids)
            save_name_cell_ids = os.path.join(save_dir, "cell_ids.npy")
            np.save(save_name_cell_ids, cell_ids)

        if args.prt_inference:
            cell_ids = np.array(test_dataset.cell_ids)


            print(str(list(cell_ids)).replace("'",""))
            print(prs_crate_all.cpu().numpy().shape)
            exit()
        
    
    def prt_metric(crates, prs, gts, criterion):
        prs_crate_all = []
        gts_crate_all = []
        for crate in crates:
            if crate not in prs: continue
            pr = prs[crate]
            gt = gts[crate]
            prs_crate_all.append(pr)
            gts_crate_all.append(gt)
            metric  = criterion(pr, gt)
            if isinstance(metric, torch.Tensor):
                metric = metric.item()
            # print(metric, end=',')
        prs_crate_all = torch.cat(prs_crate_all)
        gts_crate_all = torch.cat(gts_crate_all)
        metric_all  = criterion(prs_crate_all, gts_crate_all)
        if isinstance(metric_all, torch.Tensor):
            metric_all = metric_all.item()
        print(metric_all, end=',')

    # print("+++" * 20)
    cfg = os.path.basename(args.cfg).replace(".yaml", "")
    for data_type in prs_parsing.keys():
        prs = prs_parsing[data_type]
        gts = gts_parsing[data_type]
        
        # print("cfg,data_type,epoch", end=',')
        # metrics = ['R2','MAE','MSE','RMSE']
        # for i, metric in enumerate(metrics):
        #     end_str = ',' if i < len(metrics)-1 else '\n'
        #     num_crate = len([crate for crate in crate_parser.crates if crate in prs])
        #     print(metric+','*num_crate, end=end_str)
            
        print("{},{},{}".format(cfg, data_type, ckp['epoch']), end=',')
        # print R2
        prt_metric(crate_parser.crates, prs, gts, criterion.criterion["R2"])
        # print MAE
        prt_metric(crate_parser.crates, prs, gts, criterion.criterion["MAE"])
        # print MSE
        prt_metric(crate_parser.crates, prs, gts, criterion.criterion["MSE"])
        # print MSE
        prt_metric(crate_parser.crates, prs, gts, criterion.criterion["RMSE"])
        print()


        # # draw R2
        # prs_crate_all = []
        # gts_crate_all = []
        # for crate in crate_parser.crates:
        #     if crate not in prs: continue
        #     pr = prs[crate]
        #     gt = gts[crate]
        #     prs_crate_all.append(pr)
        #     gts_crate_all.append(gt)
        #     # print(metric, end=',')
        # prs_crate_all = torch.cat(prs_crate_all)
        # gts_crate_all = torch.cat(gts_crate_all)
        # R2 = criterion.criterion["R2"](prs_crate_all, gts_crate_all)
        # # draw
        # import matplotlib.pyplot as plt
        # save_name = os.path.join('gt-pred_ours.png')
        # fig = plt.figure(figsize=(5, 5))
        # plt.scatter(gts_crate_all, prs_crate_all)
        # xs = np.arange(gts_crate_all.min(), gts_crate_all.max(), 
        #                (gts_crate_all.max()-gts_crate_all.min())/500)
        # plt.scatter(xs, xs, s=0.15, c='black')
        # plt.title("Test R2: {:.3f}".format(R2))
        # plt.xlabel("Ground Truth")
        # plt.ylabel("Prediction")
        # plt.savefig(save_name, dpi=500)
        # plt.clf() 
        # print("... saved at", save_name)
               