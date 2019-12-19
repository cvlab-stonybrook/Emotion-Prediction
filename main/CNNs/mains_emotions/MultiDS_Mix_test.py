# Copyright (c) 2019 Zijun Wei, Yevheniia Soroka
# Licensed under the MIT License.
# Authors: Zijun Wei, Yevheniia Soroka 
# Usage(TODO): modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Email: hzwzijun@gmail.com, ysoroka@cs.stonybrook.edu
# Created: 25/Jun/2019 11:09

import os, sys
#project_root = os.path.join(os.path.expanduser('~'), 'Projects/MetaEmotion')
#project_root = os.path.join("D:/Eugenia/Emotion/MetaEmotion-master/")
project_root = os.path.join("/nfs/bigfovea/add_disk0/eugenia/Emotion/MetaEmotion-master/")
sys.path.append(project_root)

import argparse
import random
import shutil
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import CNNs.models as models
import CNNs.utils.util as CNN_utils
from torch.optim import lr_scheduler
from CNNs.dataloaders.utils import none_collate
from PyUtils.file_utils import get_date_str, get_dir, get_stem
from PyUtils import log_utils
import CNNs.dataset_loaders as custom_datasets
from CNNs.utils.config import parse_config
import torch.nn.functional as F
from CNNs.losses.z_loss import MclassCrossEntropyLoss
from CNNs.models.resnet import load_state_dict
import torchnet
from torchnet.meter import mAPMeter
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_instance(module, name, args):
    return getattr(module, name)(args)


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Pytorch Image CNN training from Configure Files")
    parser.add_argument('--config_file', required=True, help="This scripts only accepts parameters from Json files")
    input_args = parser.parse_args()

    config_file = input_args.config_file

    args = parse_config(config_file)
    if args.name is None:
        args.name = get_stem(config_file)

    torch.set_default_tensor_type('torch.FloatTensor')

    args.script_name = get_stem(__file__)
    current_time_str = get_date_str()
    if args.resume is None:
        if args.save_directory is None:
            save_directory = get_dir(os.path.join(project_root, 'ckpts', '{:s}'.format(args.name), '{:s}-{:s}'.format(args.ID, current_time_str)))
        else:
            save_directory = get_dir(os.path.join(project_root, 'ckpts', args.save_directory))
    else:
        if args.save_directory is None:
            save_directory = os.path.dirname(args.resume)
        else:
            current_time_str = get_date_str()
            save_directory = get_dir(os.path.join(args.save_directory, '{:s}'.format(args.name),
                                              '{:s}-{:s}'.format(args.ID, current_time_str)))
    print("Save to {}".format(save_directory))
    log_file = os.path.join(save_directory, 'log-{0}.txt'.format(current_time_str))
    logger = log_utils.get_logger(log_file)
    log_utils.print_config(vars(args), logger)


    print_func = logger.info
    print_func('ConfigFile: {}'.format(config_file))
    args.log_file = log_file

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    #args.distributed = args.world_size > 1
    args.distributed = False
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    num_datasets = args.num_datasets
    # model_list = [None for x in range(num_datasets)]
    # for j in range(num_datasets):
    if args.pretrained:
        print_func("=> using pre-trained model '{}'".format(args.arch))
        model= models.__dict__[args.arch](pretrained=True, num_classes=args.class_len)
    else:
        print_func("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False, num_classes=args.class_len)


    if args.freeze:
        model = CNN_utils.freeze_all_except_fc(model)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    #elif args.distributed:
    #    model.cuda()
    #    model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # # define loss function (criterion) and optimizer
    # # # Update: here
    # # config = {'loss': {'type': 'simpleCrossEntropyLoss', 'args': {'param': None}}}
    # # criterion = get_instance(loss_funcs, 'loss', config)
    # # criterion = criterion.cuda(args.gpu)
    #
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda(args.gpu)
    # criterion = MclassCrossEntropyLoss().cuda(args.gpu)

    # params = list()
    # for j in range(num_datasets):
    #     params += list(model_list[j].parameters())

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.lr_schedule:
        print_func("Using scheduled learning rate")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, [int(i) for i in args.lr_schedule.split(',')], gamma=0.1)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience)


    if args.resume:
        if os.path.isfile(args.resume):
            print_func("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            import collections
            if not args.evaluate:
                if isinstance(checkpoint, collections.OrderedDict):
                    load_state_dict(model, checkpoint, exclude_layers=['fc.weight', 'fc.bias'])


                else:
                    load_state_dict(model, checkpoint['state_dict'], exclude_layers=['module.fc.weight', 'module.fc.bias'])
                    print_func("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
            else:
                if isinstance(checkpoint, collections.OrderedDict):
                    load_state_dict(model, checkpoint, strict=True)
                else:
                    load_state_dict(model, checkpoint['state_dict'], strict=True)
                    print_func("=> loaded checkpoint '{}' (epoch {})"
                               .format(args.resume, checkpoint['epoch']))
        else:
            print_func("=> no checkpoint found at '{}'".format(args.resume))
            return



    cudnn.benchmark = True

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_func("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))


    # Data loading code
    test_loaders = [None for _ in range(num_datasets)]
    for k in range(num_datasets):
        args.ind = k

        if hasattr(args, 'test_files') and hasattr(args, 'test_loader'):
            test_dataset = get_instance(custom_datasets, args.test_loader, args)
            test_loaders[args.ind] = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, collate_fn=none_collate)
        else:
            test_loaders[args.ind] = None

        if args.evaluate:
            validate(test_loaders[args.ind], model, criterion, args, print_func, args.ind)



def train(train_loads_iter, train_loaders, model, criterion, optimizer, epoch, args, print_func):
    batch_time = CNN_utils.AverageMeter()
    data_time = CNN_utils.AverageMeter()
    losses = CNN_utils.AverageMeter()
    mAPs = mAPMeter()

    # switch to train mode
    model.train()
    if args.fix_BN:
        CNN_utils.fix_BN(model)

    batch_iters = math.ceil(args.num_iter / args.batch_size)
    for i in range(batch_iters):
        start = time.time()
        l_loss = []
        #allloss_var = 0

        optimizer.zero_grad()
        for ds in range(args.num_datasets):
            args.ind = ds

            kout = args.topX or args.class_len[args.ind] // 2

            end = time.time()
            try:
                (input, target) = train_loads_iter[ds].next()
            except StopIteration:
                train_loads_iter[ds] = iter(train_loaders[ds])
                (input, target) = train_loads_iter[ds].next()

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)
            # target_idx = target.nonzero() [:,1]
            # if torch.max(target) >= 0 and torch.max(target) < args.class_len[args.ind]:
                # compute output
            #print("Input shape {}".format(input.shape))
            output = model(input)
            output_i = output[args.ind]
            #print("Output_i device {}".format(output_i.device))
            loss = criterion(output_i, target)
            l_loss.append(loss.item())
            #allloss_var += loss
            loss.backward()

            APs.append(output_i.detach(), target)


        losses.update(sum(l_loss), input.size(0))
        APs.update(sum(APs) / len(APs), input.size(0))

        #optimizer.zero_grad()
        #allloss_var.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - start)

        if i % args.print_freq == 0:
            print_func('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'mAP {topX.val:.3f} ({topX.avg:.3f})'.format(
                   epoch, i, batch_iters, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, topX=APs))


def validate(val_loader, model, criterion, args, print_func, ind, phase='Validation'):
    if val_loader is None:
        return  0, 0
    batch_time = CNN_utils.AverageMeter()
    kl_divs = CNN_utils.AverageMeter()
    #mAPs = mAPMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            output_i = F.log_softmax(output[ind]/10.)

            kl_divs.update(F.kl_div(output_i.detach().double(), target.double(), reduction='batchmean').item(), input.size(0))
            #mAPs.add(F.softmax(output_i.detach()), target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print("pred: ", F.softmax(output[ind]/10.))
                print("true: ", target)
                print_func('[{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Dataset no. {ind}\t'
                      'dists {topX.val:.3f} ({topX.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, ind=ind,
                       topX=kl_divs))

        print_func('{phase} * dists {top1.avg:.3f}'
              .format(phase=phase, top1=kl_divs))

    return kl_divs.avg


if __name__ == '__main__':
    main()

