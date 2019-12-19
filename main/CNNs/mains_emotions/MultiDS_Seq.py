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

    args.distributed = args.world_size > 1

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
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
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



    '''
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
    '''


    cudnn.benchmark = True

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_func("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))


    # Data loading code
    val_loaders = [None for x in range(num_datasets)]
    test_loaders = [None for x in range(num_datasets)]
    train_loaders = [None for x in range(num_datasets)]
    for k in range(num_datasets):
        args.ind = k

        val_dataset = get_instance(custom_datasets, args.val_loader, args)
        if val_dataset is None:
            val_loaders[args.ind] = None
        else:
            val_loaders[args.ind] = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, collate_fn=none_collate)

        if hasattr(args, 'test_files') and hasattr(args, 'test_loader'):
            test_dataset = get_instance(custom_datasets, args.test_loader, args)
            test_loaders[args.ind] = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, collate_fn=none_collate)
        else:
            # test_dataset = None
            test_loaders[args.ind] = None

        #if args.evaluate:
        #    validate(test_loaders[args.ind], model_list[k], criterion, args, print_func)
        #    return
        # if not args.evaluate: #else:
        #     train_samplers = [None for x in range(num_datasets)]
        #     train_dataset = get_instance(custom_datasets, args.train_loader, args)
        #
        #     if args.distributed:
        #         train_samplers[args.ind] = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #     else:
        #         train_samplers[args.ind] = None
        #
        #     train_loaders[args.ind] = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=args.batch_size, shuffle=(train_samplers[args.ind] is None),
        #         num_workers=args.workers, pin_memory=True, sampler=train_samplers[args.ind], collate_fn=none_collate)
        if not args.evaluate: #else:
            # train_samplers = [None for x in range(num_datasets)]
            train_dataset = get_instance(custom_datasets, args.train_loader, args)

            if args.distributed:
                train_samplers = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_samplers = None

            train_loaders[args.ind] = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=train_samplers is None,
                num_workers=args.workers, pin_memory=True, sampler=train_samplers, collate_fn=none_collate)
    # TRAINING
    # Iterate through all datasets
    best_prec1 = [-1 for _ in range(num_datasets)]
    is_best = [None for _ in range(num_datasets)]
    for k in range(num_datasets):
        args.ind = k

        if args.lr_schedule:
            print_func("Using scheduled learning rate")
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, [int(i) for i in args.lr_schedule.split(',')], gamma=0.1)
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=args.lr_patience)
        if args.ind > 0:
        #     load_state_dict(model_list[args.ind], model_list[args.ind-1].state_dict())
            for param_group in optimizer.param_groups:
                 param_group['lr'] = args.lr

        print_func("Starting training on dataset {}".format(args.ind))

        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
                train_samplers[args.ind].set_epoch(epoch)
            if args.lr_schedule:
                # CNN_utils.adjust_learning_rate(optimizer, epoch, args.lr)
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print_func("Epoch: [{}], learning rate: {}".format(epoch, current_lr))

            # train for one epoch
            train(train_loaders[args.ind], model, criterion, optimizer, epoch, args, print_func)

            # evaluate and save
            val_prec1 = [None for x in range(num_datasets)]
            test_prec1 = [None for x in range(num_datasets)]
            for j in range(num_datasets):
                # if j != args.ind:
                #     load_state_dict(model_list[j], model_list[args.ind].state_dict())
                # evaluate on validation set
                if val_loaders[j]:
                    val_prec1[j], _ = validate(val_loaders[j], model, criterion, args, print_func, j)
                else:
                    val_prec1[j] = 0
                # remember best prec@1 and save checkpoint
                is_best[j] = val_prec1[j] > best_prec1[j]
                best_prec1[j] = max(val_prec1[j], best_prec1[j])

                if is_best[j]:
                    save_ind = j
                else:
                    save_ind="#"
                CNN_utils.save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_prec1': best_prec1[j],
                    'optimizer' : optimizer.state_dict(),
                }, is_best[j], file_directory=save_directory, epoch=epoch, save_best_only=args.save_best_only, ind=save_ind)

                test_prec1[j], _ = validate(test_loaders[j], model, criterion, args, print_func, j, phase='Test')

            print_func("Val precisions: {}".format(val_prec1))
            print_func("Test precisions: {}".format(test_prec1))

        print_func("Finished training on dataset {}".format(args.ind))



def train(train_loader, model, criterion, optimizer, epoch, args, print_func):
    batch_time = CNN_utils.AverageMeter()
    data_time = CNN_utils.AverageMeter()
    losses = CNN_utils.AverageMeter()
    top1 = CNN_utils.AverageMeter()
    topX = CNN_utils.AverageMeter()

    kout = args.topX or args.class_len[args.ind] // 2

    # switch to train mode
    model.train()
    if args.fix_BN:
        CNN_utils.fix_BN(model)
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        # target_idx = target.nonzero() [:,1]
        # if torch.max(target) >= 0 and torch.max(target) < args.class_len[args.ind]:
            # compute output
        output = model(input)


        output_i = output[args.ind]

        loss = criterion(output_i, target)

        losses.update(loss.item(), input.size(0))

        prec1, precX = CNN_utils.accuracy(output_i, target, topk=(1, kout))

        top1.update(prec1.item(), input.size(0))
        topX.update(precX.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            print_func('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@{kout} {topX.val:.3f} ({topX.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, kout=kout, topX=topX))


def validate(val_loader, model, criterion, args, print_func, ind, phase='Validation'):
    if val_loader is None:
        return  0, 0
    batch_time = CNN_utils.AverageMeter()
    losses = CNN_utils.AverageMeter()
    top1 = CNN_utils.AverageMeter()
    topX = CNN_utils.AverageMeter()

    kout = args.topX or args.class_len[ind] // 2

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
            output = output[ind]

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            prec1, precX = CNN_utils.accuracy(output, target, topk=(1, kout))

            top1.update(prec1.item(), input.size(0))
            topX.update(precX.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_func('[{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Dataset no. {ind}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@{kout} {topX.val:.3f} ({topX.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, ind=ind, loss=losses,
                       top1=top1, kout=kout, topX = topX))

        print_func('{phase} * Prec@1 {top1.avg:.3f}'
              .format(phase=phase, top1=top1))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()

