import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import pickle

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model, classifier, dataset, args):
    # Get labeled data list
    if args.use_unlabeled:
        lb_file = args.train_lb_list
    else:
        lb_file = args.train_list
    with open(lb_file, 'r') as f:
        lb_list = f.read().split('\n')[:-1]

    if args.cls_prediction != '':
        print(f"Read Class Predictions... ({args.cls_prediction})")
        with open(args.cls_prediction, 'rb') as f:
            ext_pred = pickle.load(f)
    else:
        ext_pred = None

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    # verbose interval
    if len(databin) // 20 > 0:
        interval = len(databin) // 20 
    else:
        interval = 1

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        classifier.cuda()

        lb_cnt, ulb_cnt = 0, 0
        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0] # 20xx_00xxxx
            label = pack['label'][0]
            size = pack['size']
            
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

            # [20, H, W]
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0]
                 for o in outputs]), 0)

            highres_cams = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            # [20, H, W]
            highres_cam = torch.sum(torch.stack(highres_cams, 0), 0)[:, 0, :size[0], :size[1]]
            
            # Use class prediction(Ulb, val) - 
            if not (img_name in lb_list):
                # External Models
                if ext_pred is not None and img_name in ext_pred['idx']:
                    idx = ext_pred['idx'][img_name]
                    pred = ext_pred['pred'][idx]
                    label = ext_pred['pred'][idx]

                # Trained Model (at train_cam)
                else:
                    # Multi-scale Ensemble(Option 1. Sum logits)
                    preds = [classifier(img[0].cuda(non_blocking=True)) for img in pack['img']]
                    pred = torch.sum(torch.cat(preds, 0), 0)
                    pred = torch.sigmoid(pred)
                    
                    # Replace label into prediction
                    label = pred >= 0.5
                
                # Zero-predicted
                if torch.sum(label) == 0:
                    # remain max class
                    label = pred >= torch.max(pred)
                label = label.detach().cpu()

                ulb_cnt += 1
            else:
                lb_cnt += 1

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % interval == 0:
                print("%d " % ((5*iter+1)//interval), end='')
        
        print('lb:', lb_cnt, 'ulb:', ulb_cnt)


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()
    classifier = getattr(importlib.import_module(args.cam_network), 'Net')()
    classifier.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    classifier.eval()

    n_gpus = torch.cuda.device_count()

    print('Train List:', args.train_list)
    print('Infer List:', args.infer_list)
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    # Add infer_list dataset
    if args.train_list != args.infer_list:
        dataset_aug = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                            voc12_root=args.voc12_root, scales=args.cam_scales)
        dataset = ConcatDataset([dataset, dataset_aug]) 

    print(f'{len(dataset)} Images.')


    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, classifier, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
