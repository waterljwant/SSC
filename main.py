#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
python script to train the SSC model
---
Jie Li
jieli_cn@163.com
Nanjing University of Science and Technology
Aug 25, 2019
"""


from utils.seed import seed_torch
import os

import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable


import datetime

from dataloaders import make_data_loader
import sscMetrics

from models import make_model
import config


parser = argparse.ArgumentParser(description='PyTorch SSC Training')
parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'nyucad', 'debug'],
                    help='dataset name (default: nyu)')
parser.add_argument('--model', type=str, default='ddrnet', choices=['ddrnet', 'aicnet', 'grfnet', 'palnet'],
                    help='model name (default: palnet)')
# parser.add_argument('--data_augment', default=False, type=bool,  help='data augment for training')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_adj_n', default=100, type=int, metavar='LR', help='every n epochs adjust learning rate once')
parser.add_argument('--lr_adj_rate', default=0.1, type=float, metavar='LR', help='scale while adjusting learning rate')

parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', default='./', metavar='DIR', help='path to checkpoint')
# parser.add_argument('--logdir', default='./logs_debug', metavar='DIR', help='path to logs')

parser.add_argument('--model_name', default='SSC_debug', type=str, help='name of model to save check points')

# parser.add_argument('--w', default=0.05, type=float, help='weight')


global args
args = parser.parse_args()

seed_torch(2019)


def main():
    # ---- Check CUDA
    if torch.cuda.is_available():
        print("Great, You have {} CUDA device!".format(torch.cuda.device_count()))
    else:
        print("Sorry, You DO NOT have a CUDA device!")

    train_time_start = datetime.datetime.now()
    train()
    print('Training finished in: {}'.format(datetime.datetime.now() - train_time_start))


def train():
    # ---- create model ---------- ---------- ---------- ---------- ----------#
    net = make_model(args.model, num_classes=12).cuda()
    net = torch.nn.DataParallel(net)  # Multi-GPU

    # ---- optionally resume from a checkpoint --------- ---------- ----------#
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            cp_states = torch.load(args.resume)
            net.load_state_dict(cp_states['state_dict'], strict=True)
        else:
            raise Exception("=> NO checkpoint found at '{}'".format(args.resume))

    # -------- ---------- --- Set checkpoint --------- ---------- ----------#
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H.%M.%S")
    # model_info = 'epoch{}_lr{}'.format(args.epochs, args.lr)
    cp_filename = args.checkpoint + 'cp_{}.pth.tar'.format(args.model_name)
    cp_best_filename = args.checkpoint + 'cpBest_{}.pth.tar'.format(args.model_name)

    # ---- Define loss function (criterion) and optimizer ---------- ----------#
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss(weight=config.class_weights, ignore_index=255).cuda()

    # ---- Print Settings for training -------- ---------- ---------- ----------#
    print('Training epochs:{} \nInitial Learning rate:{} \nBatch size:{} \nNumber of workers:{}'.format(
        args.epochs,
        args.lr,
        args.batch_size,
        args.workers,
        cp_filename))
    print("Checkpoint filename:{}".format(cp_filename))

    # ---- Data loader
    train_loader, val_loader = make_data_loader(args)

    np.set_printoptions(precision=1)

    # ---- Train
    step_count_all = 0
    best_miou = 0

    print("Start training")

    for epoch in range(0, args.epochs):
        # print("epoch {}".format(epoch))
        net.train()  # switch to train mode
        # adjust_learning_rate(optimizer, args.lr, epoch, n=args.lr_adj_n, rate=args.lr_adj_rate)  # n=10, rate=0.9
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_adj_n, gamma=args.lr_adj_rate, last_epoch=-1)
        
        decs_str = 'Training epoch {}/{}'.format(epoch + 1, args.epochs)
        log_loss_1epoch = 0.0
        step_count = 0

        torch.cuda.empty_cache()
        for step, (rgb, depth, tsdf, target, position, _) in tqdm(enumerate(train_loader), desc=decs_str, unit='step'):
            # target should be a LongTensor. (bs, 60L, 36L, 60L)
            y_true = target.long().contiguous()
            y_true = Variable(y_true.view(-1)).cuda()  # bs * D * H * W

            # ---- (bs, C, D, H, W), channel first for Conv3d in pyTorch
            # FloatTensor to Variable. (bs, channels, 240L, 144L, 240L)
            x_depth = Variable(depth.float()).cuda()
            position = position.long().cuda()

            if args.model == 'palnet':
                x_tsdf = Variable(tsdf.float()).cuda()
                y_pred = net(x_depth=x_depth, x_tsdf=x_tsdf, p=position)
            else:
                x_rgb = Variable(rgb.float()).cuda()
                y_pred = net(x_depth=x_depth, x_rgb=x_rgb, p=position)

            y_pred = y_pred.permute(0, 2, 3, 4, 1).contiguous()  # (BS, C, D, H, W) --> (BS, D, H, W, C)
            y_pred = y_pred.view(-1, 12)  # C = 12

            optimizer.zero_grad()
            loss = loss_func(y_pred, y_true)

            loss.backward()
            optimizer.step()

        # ---- Evaluate on validation set
        v_prec, v_recall, v_iou, v_acc, v_ssc_iou, v_mean_iou = validate_on_dataset_stsdf(net, val_loader)
        print('Validate with TSDF:epoch {}, p {:.1f}, r {:.1f}, IoU {:.1f}'.format(epoch + 1, v_prec*100.0, v_recall*100.0, v_iou*100.0))
        print('pixel-acc {:.4f}, mean IoU {:.1f}, SSC IoU:{}'.format(v_acc*100.0, v_mean_iou*100.0, v_ssc_iou*100.0))

        # ---- Save Checkpoint
        is_best = v_mean_iou > best_miou
        best_miou = max(v_mean_iou, best_miou)
        state = {'state_dict': net.state_dict()}
        torch.save(state, cp_filename)
        if is_best:
            print('Yeah! Got better mIoU {}% in epoch {}. State saved'.format(100.0*v_mean_iou, epoch + 1))
            torch.save(state, cp_best_filename)  # Save Checkpoint

# --------------------------------------------------------------------------------------------------------------


def validate_on_dataset_stsdf(model, date_loader, save_ply=False):
    """
    Evaluate on validation set.
        model: network with parameters loaded
        date_loader: TEST mode
    """
    model.eval()  # switch to evaluate mode.
    val_acc, val_p, val_r, val_iou = 0.0, 0.0, 0.0, 0.0
    _C = 12
    val_cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
    val_iou_ssc = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
    count = 0
    with torch.no_grad():
        # ---- STSDF  depth, input, target, position, _
        for step, (rgb, depth, volume, y_true, nonempty, position, filename) in tqdm(enumerate(date_loader), desc='Validating', unit='frame'):
            var_x_depth = Variable(depth.float()).cuda()
            position = position.long().cuda()

            if args.model == 'palnet':
                var_x_volume = Variable(volume.float()).cuda()
                y_pred = model(x_depth=var_x_depth, x_tsdf=var_x_volume, p=position)
            else:
                var_x_rgb = Variable(rgb.float()).cuda()
                y_pred = model(x_depth=var_x_depth, x_rgb=var_x_rgb, p=position)  # y_pred.size(): (bs, C, W, H, D)

            y_pred = y_pred.cpu().data.numpy()  # CUDA to CPU, Variable to numpy
            y_true = y_true.numpy()  # torch tensor to numpy
            nonempty = nonempty.numpy()

            p, r, iou, acc, iou_sum, cnt_class = validate_on_batch(y_pred, y_true, nonempty)
            count += 1
            val_acc += acc
            val_p += p
            val_r += r
            val_iou += iou
            val_iou_ssc = np.add(val_iou_ssc, iou_sum)
            val_cnt_class = np.add(val_cnt_class, cnt_class)
            # print('acc_w, acc, p, r, iou', acc_w, acc, p, r, iou)

    val_acc = val_acc / count
    val_p = val_p / count
    val_r = val_r / count
    val_iou = val_iou / count
    val_iou_ssc, val_iou_ssc_mean = sscMetrics.get_iou(val_iou_ssc, val_cnt_class)
    return val_p, val_r, val_iou, val_acc, val_iou_ssc, val_iou_ssc_mean


def validate_on_batch(predict, target, nonempty=None):  # CPU
    """
        predict: (bs, channels, D, H, W)
        target:  (bs, channels, D, H, W)
    """
    # TODO: validation will increase the usage of GPU memory!!!
    y_pred = predict
    y_true = target
    p, r, iou = sscMetrics.get_score_completion(y_pred, y_true, nonempty)
    #acc, iou_sum, cnt_class = sscMetrics.get_score_semantic_and_completion(y_pred, y_true, stsdf)
    acc, iou_sum, cnt_class, tp_sum, fp_sum, fn_sum = sscMetrics.get_score_semantic_and_completion(y_pred, y_true, nonempty)
    # iou = np.divide(iou_sum, cnt_class)
    return p, r, iou, acc, iou_sum, cnt_class


# static method
def adjust_learning_rate(optimizer, lr, epoch, n=10, rate=0.9):
    """Sets the learning rate to the initial LR decayed by rate=0.9 every n=10 epochs"""
    new_lr = lr * (rate ** (epoch // n))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    if epoch % n == 0:
        print('Current learning rate is: {}'.format(new_lr))


if __name__ == '__main__':
    main()
