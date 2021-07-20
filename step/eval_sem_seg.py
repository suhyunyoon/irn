
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

#from torchvision.datasets import Cityscapes
from chainercv.datasets import CityscapesSemanticSegmentationDataset

import imageio

def run(args):
    if args.dataset == 'voc12':
        dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
        ids = dataset.ids
    elif args.dataset == 'cityscapes':
        dataset = CityscapesSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.cityscapes_root, label_resolution=args.cityscapes_mode)
        ids = list(map(lambda p: os.path.splitext(os.path.basename(p))[0], dataset.img_paths))
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    
    preds = []
    for id in ids:
        cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})
