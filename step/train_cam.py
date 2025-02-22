
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils, metrics


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')
    val_acc_meter = pyutils.AverageMeter('acc1', 'acc2')
    val_map_meter = pyutils.AverageMeter('map1', 'map2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)
            # metrics
            acc1, _, map1 = metrics.eval_multilabel_metric(label.detach().cpu(), x.detach().cpu())

            val_loss_meter.add({'loss1': loss1.item()})
            val_acc_meter.add({'acc1': acc1.item()})
            val_map_meter.add({'map1': map1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')), end=' ')
    print('acc: %.4f' % (val_acc_meter.pop('acc1')), end=' ') 
    print('map: %.4f' % (val_map_meter.pop('map1'))) 

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()

    if args.use_unlabeled:
        train_cam_list = args.train_lb_list
    else:
        train_cam_list = args.train_list

    print('Train List:', train_cam_list)
    print('Val List:', args.val_list)

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_cam_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")

    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print(f'Train dataset: {len(train_dataset)}')
    print(f'Val dataset: {len(val_dataset)}')

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):
        verbose_flag = False
        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0 or not verbose_flag:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                verbose_flag = True

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()
