
import builtins
import torch.distributed as dist
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
import numpy as np
import math
from scipy.io import loadmat
import model.ResNet as models
from model.CaCo import CaCo, CaCo_PN
from ops.os_operation import mkdir, mkdir_rank
from training.train_utils import adjust_learning_rate2,save_checkpoint
from data_processing.loader import TwoCropsTransform2,GaussianBlur,Solarize
from data_processing.datasets import get_dataset,Hyper2X
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=Warning)


# 按多个参数分子文件夹存储模型文件和log文件
def init_log_path(args,batch_size):
    save_path = os.path.join(os.getcwd(), args.log_path)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, args.dataset)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "Type_"+str(args.type))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "lr_" + str(args.lr) + "_" + str(args.lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memlr_"+str(args.memory_lr) +"_"+ str(args.memory_lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "t_" + str(args.moco_t) + "_memt" + str(args.mem_t))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "wd_" + str(args.weight_decay) + "_memwd" + str(args.mem_wd)) 
    mkdir_rank(save_path,args.rank)
    if args.moco_m_decay:
        save_path = os.path.join(save_path, "mocomdecay_" + str(args.moco_m))
    else:
        save_path = os.path.join(save_path, "mocom_" + str(args.moco_m))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memgradm_" + str(args.mem_momentum))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "hidden" + str(args.mlp_dim)+"_out"+str(args.moco_dim))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "batch_" + str(batch_size))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "epoch_" + str(args.epochs))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "warm_" + str(args.warmup_epochs))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "time_" + str(args.time))
    mkdir_rank(save_path,args.rank)
    return save_path


def main_worker(gpu, ngpus_per_node, args):
    params = vars(args)
    args.gpu = gpu
    init_lr = args.lr
    total_batch_size = args.batch_size
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.dataset=='IndianPines' or args.dataset=='CongHoa' or args.dataset=='DongXing':
        print("加载" + args.dataset + "数据集！")
        DATASET = args.dataset
        FOLDER = args.data_folder
        LOAD_DATA = args.load_data
        hyperparams = vars(args)
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
        img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
        N_CLASSES = len(LABEL_VALUES)
        N_BANDS = img.shape[-1]
        hyperparams.update(
            {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
        print("训练集加载比例：",LOAD_DATA)
        if LOAD_DATA:
            train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
            test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
            train_gt = np.load(train_gt_file, 'r')
        else:
            print("已加载全部可用训练样本！")
            train_gt = gt

        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        print("类别：{}".format(mask))
        print("训练集每类个数{}".format(tmp))
        print("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
        train_dataset = Hyper2X(img, train_gt, **hyperparams)
        print(args.dataset + "数据集加载完毕!")
    else:
        print("We don't support this dataset currently")
        exit()

    # 新建Memory Bank
    Memory_Bank = CaCo_PN(args.cluster,args.moco_dim)
    # 新建model
    model = CaCo(models.__dict__[args.arch], args, args.moco_dim, args.moco_m, N_BANDS) 
    from model.optimizer import  LARS
    optimizer = LARS(model.parameters(), init_lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        Memory_Bank=Memory_Bank.cuda(args.gpu)
        # Memory_Bank_upper=Memory_Bank_upper.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        model.cuda()
        Memory_Bank.cuda()
        # Memory_Bank_upper.cuda()
        print("Only DistributedDataParallel is supported.")
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    save_path = init_log_path(args,total_batch_size)
    print("save_path: ", save_path)
    if not args.resume:
        args.resume = os.path.join(save_path,"checkpoint_best.pth.tar")
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Memory_Bank.load_state_dict(checkpoint['Memory_Bank'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    model.eval()
    # 初始化 memory bank
    if args.ad_init and not os.path.isfile(args.resume):
        from training.init_memory import init_memory
        # init_memory(train_loader, model, Memory_Bank, criterion,
        #         optimizer, 0, args)
        init_memory(train_loader, model, Memory_Bank, criterion,
                optimizer, 0, args)
        print("初始化memory bank完成!")
    
    knn_path = os.path.join(save_path,"knn.log")
    train_log_path = os.path.join(save_path,"train.log")
    best_Acc=0

    # 模型预训练，遍历所有epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate2(optimizer, epoch, args, init_lr)    
        if args.moco_m_decay:
            moco_momentum = adjust_moco_momentum(epoch, args)
        else:
            moco_momentum = args.moco_m
        from training.train_caco import train_caco
        acc1 = train_caco(train_loader, model, Memory_Bank, criterion,
                                optimizer, epoch, args, train_log_path,moco_momentum)  
        is_best=best_Acc>acc1
        best_Acc=max(best_Acc,acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):
            save_dict={
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc':best_Acc,
            # 'knn_acc': knn_test_acc,
            'knn_acc': 3.14,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'Memory_Bank':Memory_Bank.state_dict(),
            }

            if epoch%10==9:
                tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
            tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')
            save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
    print("模型预训练完成!")
           
        
def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    return 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)

