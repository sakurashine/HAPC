
import argparse
import builtins
import math
import os
import random
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import model.ResNet_Linear as models
import numpy as np
from ops.os_operation import mkdir
from data_processing.datasets import get_dataset, HyperX
from data_processing.utils import  get_device, sample_gt, count_sliding_window, compute_imf_weights, metrics, logger, display_goundtruth, sliding_window, grouper
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import visdom
import datetime
import logging
from scipy.io import loadmat


parser = argparse.ArgumentParser(description='PyTorch HSI Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=16, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# additional configs:
parser.add_argument('--pretrained', default='./train_log/CongHoa/Type_0/lr_0.3_0.02/memlr_0.3_0.02/t_0.08_memt0.08/wd_1.5e-06_memwd0/mocomdecay_0.99/memgradm_0.9/hidden128_out128/batch_32/epoch_1/warm_5/time_1/checkpoint_best.pth.tar', type=str,
                    help='path to CaCo pretrained checkpoint')
parser.add_argument("--dataset", type=str, default="CongHoa", help="which dataset is used to finetune")
parser.add_argument('--folder', default="./dataset/CongHoa", type=str, metavar='DIR',
                        help='path to dataset')
parser.add_argument("--load_data", type=str, default=200, 
                           help="Samples use of training")
parser.add_argument('--training_percentage', type=float, default=0.10, 
                           help="Percentage of samples to use for training")
parser.add_argument('--sampling_mode', type=str, default='random',
                           help="Sampling mode (random sampling or disjoint, default: random)")
parser.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
parser.add_argument('--patch_size', type=int, default=15,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
parser.add_argument('--validation_percentage', type=float, default=0.2,
                           help="In the training data set, percentage of the labeled data are randomly "
                                "assigned to validation groups.")
parser.add_argument('--supervision', type=str, default='full',
                         help="full supervision or semi supervision ")    
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")   
parser.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")  
parser.add_argument('--sample_nums', type=int, default=None,
                           help="Number of samples to use for training and validation") 
parser.add_argument('--run', type=int, default=1,
                    help="Running times.")
parser.add_argument('--save_epoch', type=int, default=5,
                         help="Training save epoch")
parser.add_argument('--fine_tune', type=str, default='no',
                         help="Choose linear prob or fine-tune")    
parser.add_argument('--desc', type=str, default=' ',
                         help="Describing current experiment with one word")     
parser.add_argument('--raw', type=str, default='no',
                         help="Use raw image or not")                

best_acc1 = 0

def main():
    args = parser.parse_args() 
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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    RUN = args.run
    DATASET = args.dataset
    file_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
    log = logger('./test_log/logs-' + file_date + DATASET +'.txt')
    logging.getLogger('matplotlib.font_manager').disabled = True
    log.info("---------------------------------------------------------------------")
    log.info("-----------------------------Next run log----------------------------")
    log.info("---------------------------{}--------------------------".format(log_date))
    log.info("---------------------------------------------------------------------")
    CUDA_DEVICE = get_device(log, args.cuda)
    FOLDER = args.folder
    SAMPLE_NUMS = args.sample_nums
    LOAD_DATA = args.load_data
    TRAINING_PERCENTAGE = args.training_percentage
    SAMPLING_MODE = args.sampling_mode
    CLASS_BALANCING = args.class_balancing
    
    hyperparams = vars(args)
    img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
    N_CLASSES = len(LABEL_VALUES)
    N_BANDS = img.shape[-1]
    FINE_TUNE = args.fine_tune
    RAW = args.raw
    hyperparams.update(
        {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'center_pixel': True, 'device': CUDA_DEVICE, 'fine_tune': FINE_TUNE})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    log.info("已加载{}数据集".format(DATASET))
    log.info("标签类名：{}".format(LABEL_VALUES))
    log.info("标签数量：{}".format(N_CLASSES))
    
    # 定义一个列表用于存储run多次的总体准确率
    acc_dataset = np.zeros([RUN, 1])
    # 定义一个列表用于存储run多次的每类平均准确率
    A = np.zeros([RUN, N_CLASSES-1])
    # 定义一个列表用于存储run多次的总体准确率
    K = np.zeros([RUN, 1])
    
    for i in range(RUN):
        log.info("==========================================================================================")
        log.info("======================================RUN:{}===============================================".format(i))
        log.info("==========================================================================================")
        model = models.resnet18(num_classes=N_CLASSES, num_bands=N_BANDS, fine_tune = FINE_TUNE)
        for name, param in model.named_parameters():  # 对读取的model里的参数遍历，如果是非FC层的参数，则取消梯度回传
            if args.fine_tune == 'no':
                ft = False
            else:
                ft = True
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = ft
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        linear_keyword="fc"
        
        # 加载预训练模型
        if args.pretrained:
            if os.path.isfile(args.pretrained): 
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                # rename caco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("encoder_q."):]] = state_dict[k]
                        del state_dict[k]
                args.start_epoch = 0
                # model = nn.DataParallel(model).cuda()
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))  

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        init_lr = args.lr
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        from ops.LARS import SGD_LARC
        optimizer = SGD_LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)
        args.pretrained = os.path.abspath(args.pretrained)
        save_dir = os.path.split(args.pretrained)[0]
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "linear_lars")
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "bs_%d" % args.batch_size)
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "lr_%.3f" % args.lr)
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "wd_" + str(args.weight_decay))
        if args.rank==0:
            mkdir(save_dir)
        cudnn.benchmark = True

        # 数据集加载
        if LOAD_DATA:
            if DATASET == 'CongHoa' or DATASET == 'DongXing':
                log.info("采样方式：固定样本个数")
                gt_file = train_gt_file = './dataset/' + DATASET + '/' + str(LOAD_DATA) + '/' + DATASET + '_' + str(LOAD_DATA)+ '_' + str(i) + '.mat'
                train_gt = loadmat(gt_file)['TR']
                test_gt = loadmat(gt_file)['TE']
                log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
            else:
                log.info("采样方式：固定样本比例")
                train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
                test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
                train_gt = np.load(train_gt_file, 'r')
                test_gt = np.load(test_gt_file, 'r')
                log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
        else:
            log.info("采样方式：固定样本个数")
            train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode='fixed', sample_nums=SAMPLE_NUMS)
            log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
            log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
        if SAMPLING_MODE == 'fixed':
            vis = visdom.Visdom(
                env=DATASET + ' ' + hyperparams['arch'] + ' ' + 'PATCH_SIZE' + str(
                    hyperparams['patch_size']) + ',' + 'EPOCH' + str(hyperparams['epoch']))
        else:
            vis = visdom.Visdom(env=DATASET + ' ' + hyperparams['arch'] + ' ' + 'PATCH_SIZE' + str(
                hyperparams['patch_size']) + ' ' + 'EPOCH' + str(hyperparams['epochs']))
        if not vis.check_connection:
            print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
        
        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        mask = np.unique(test_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(test_gt==v))
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights).float().cuda()
        

        train_dataset = HyperX(img, train_gt, **hyperparams)       
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    shuffle=True,
                                    drop_last=False)
        log.info("Train dataloader:{}".format(len(train_loader))) # 9
        for k, v in hyperparams.items():
            log.info("{}:{}".format(k, v))
        log.info("Network :")
        # 训练分类器
        for epoch in range(args.epochs):
            learning_rate = adjust_learning_rate(optimizer, init_lr, epoch, args) 
            feature_bank, feature_labels = train(train_loader, model, criterion, optimizer, epoch, args, N_CLASSES)
        
        # 测试阶段
        prediction = test(model, img, hyperparams)
        results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
        
        color_gt = display_goundtruth(DATASET, gt=prediction, vis=vis, caption="Testing ground truth(full)" + "RUN{}".format(i+1))
        if args.load_data:
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} data{} patch{} {} finetune{} RUN{}Testing gt(full).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['load_data'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
            mask = np.zeros(gt.shape, dtype='bool')
            for l in IGNORED_LABELS:
                mask[gt == l] = True
            prediction[mask] = 0
            color_gt_raw = display_goundtruth(DATASET, gt=prediction, vis=vis, caption="Testing ground truth(semi)"+"RUN{}".format(i))
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} data{} patch{} {} finetune{} RUN{}Testing gt(semi).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['load_data'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt_raw)
        acc_dataset[i,0] = results['Accuracy']  # 把每次RUN的总体准确率保存
        A[i] = results['F1 scores'][1:]  # 把每次RUN的每类准确率保存
        K[i,0] = results['Kappa']  # 把每次RUN的Kappa准确率保存

        log.info('----------Training result----------')
        log.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
        log.info("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
        log.info("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
        log.info("\nKappa:\n{:.4f}".format(results['Kappa']))
        print("Acc_dataset {}".format(acc_dataset))

    
    # 计算多轮的平均准确率
    OA_std = np.std(acc_dataset)
    OAMean = np.mean(acc_dataset)
    AA_std = np.std(A,1)
    AAMean = np.mean(A,1)
    Kappa_std = np.std(K)
    KappaMean = np.mean(K)

    AA = list(map('{:.2f}%'.format, AAMean))
    p = []
    log.info("{}数据集的结果如下".format(DATASET))
    for item,std in zip(AAMean,AA_std):
        p.append(str(round(item*100,2))+"+-"+str(round(std,2)))
    log.info(np.array(p))
    log.info("AAMean {:.2f} +-{:.2f}".format(np.mean(AAMean)*100,np.mean(AA_std)))
    log.info("{}".format(acc_dataset))
    log.info("OAMean {:.2f} +-{:.2f}".format(OAMean,OA_std))
    log.info("{}".format(K))
    log.info("KappaMean {:.2f} +-{:.2f}".format(KappaMean,Kappa_std))
    
   

def train(train_loader, model, criterion, optimizer, epoch, args, N_CLASSES):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model = model.to('cuda')
    model.eval()
    feature = torch.rand(1,128).cuda() 
    label = torch.rand(1) .cuda()
    end = time.time()
    for i, (images, target ) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        # print(model)
        output, flat_emb = model(images)
        # output = F.normalize(output, dim=1)
        feature = torch.cat([feature, flat_emb], dim=0)
        
        label = torch.cat([label, target], dim=0)
        loss = criterion(output, target)  # torch.Size([32, 17]),torch.Size([32])

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))  # 32
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    return feature, label

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, mAP],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            output = output.to(device)
            target = target.to(device)
            # output = concat_all_gather(output)
            # target = concat_all_gather(target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)  # torch.Size([32, 17]),torch.Size([32])
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mAP {mAP.avg:.3f} '
              .format(top1=top1, top5=top5, mAP=mAP))

    return top1.avg

# 定义t-SNE绘图函数
def plot(x, colors, epoch, args):
    flatui = ["#9b59b6", "#7A80A4", "#0a5757", "#1DA96C", "#9FD06C", "#05B8E1",
            "#7F655E", "#FDA190", "#4A4D68", "#D1E0E9", "#C4C1C5", "#F2D266",
            "#B15546", "#CE7452", "#A59284", "#DFD2A3", "#F9831A"]  # 用于WHU-Hi-HanChuan数据集的可视化调色盘
    palette = np.array(sns.color_palette(flatui))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.cpu().numpy().astype(np.int8)])
    txts = []
    for i in range(17):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    if args.load_data:
        plt.savefig('./t-SNE/'+ str(datetime.date.today()) + '_' + args.dataset  + args.load_data + '_' + args.desc + '_' + args.fine_tune + ' finetune_' + 'epoch' + str(epoch) + '_tSNE.png', dpi=120)
    elif args.sample_nums:
        plt.savefig('./t-SNE/'+ str(datetime.date.today()) + '_' + args.dataset  + str(args.sample_nums) + '_' + args.desc + '_' + args.fine_tune + ' finetune_' + 'epoch' + str(epoch) + '_tSNE.png', dpi=120)
    return f, ax, txts


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    # probs = np.zeros(img.shape[:2] + (n_classes,))
    probs = np.zeros(img.shape[:2])
    img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'reflect')
    
    iterations = count_sliding_window(img, step=hyperparams['test_stride'], window_size=(patch_size, patch_size))
    
    for batch in tqdm(grouper(batch_size, sliding_window(img, step=1, window_size=(patch_size, patch_size))),
                      total=(iterations//batch_size),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            data = [b[0] for b in batch]

            data = np.copy(data)

            data = data.transpose(0, 3, 1, 2)

            data = torch.from_numpy(data)

            # data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]

            data = data.to(device)
            data = data.type(torch.cuda.FloatTensor)
            # print(data.shape)
            output, _ = net(data)
            # print(output.shape)
            _, output = torch.max(output, dim=1)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')
            if center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x, y] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# 动态调整学习率
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    # print("######################init_lr", init_lr)
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    # print("######################cur_lr", cur_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return param_group['lr']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred = pred.to(device)
        target = target.to(device)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()


