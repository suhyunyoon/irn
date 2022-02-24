
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion


def calc_iou(preds, labels):
    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    return iou


def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    #labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    print('Chainer eval set:', args.chainer_eval_set)
    print(f'{len(dataset)} Images.')
    # Get labeled data list
    if args.use_unlabeled:
        lb_file = args.train_lb_list
    else:
        lb_file = args.train_list
    with open(lb_file, 'r') as f:
        lb_list = f.read().split('\n')[:-1]
    #print(len(lb_list), lb_list[0], lb_list[-1])

    preds_lb, labels_lb = [], []
    preds_ulb, labels_ulb = [], []
    for i, id in enumerate(dataset.ids):
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]

        if id in lb_list: 
            preds_lb.append(cls_labels.copy())
            labels_lb.append(dataset.get_example_by_keys(i, (1,))[0])
        else:
            preds_ulb.append(cls_labels.copy())
            labels_ulb.append(dataset.get_example_by_keys(i, (1,))[0])

    if len(preds_lb) > 0:
        iou_lb = calc_iou(preds_lb, labels_lb)
        print(f'Labeled Data({len(labels_lb)}):')
        print({'iou': iou_lb, 'miou': np.nanmean(iou_lb)})

    if len(preds_ulb) > 0:
        iou_ulb = calc_iou(preds_ulb, labels_ulb)
        print(f'Unlabeled Data({len(labels_ulb)}):')
        print({'iou': iou_ulb, 'miou': np.nanmean(iou_ulb)})

    if len(preds_lb) > 0 and len(preds_ulb) > 0:
        iou = calc_iou(preds_lb + preds_ulb, labels_lb + labels_ulb)
        print(f'TOTAL({len(labels_lb)+len(labels_ulb)}):')
        print({'iou': iou, 'miou': np.nanmean(iou)})
