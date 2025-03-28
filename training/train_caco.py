import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from training.train_utils import AverageMeter, ProgressMeter, accuracy
import matplotlib.pyplot as plt
import copy
import datetime
import seaborn as sns

def train_caco(train_loader, model, Memory_Bank, criterion,
          optimizer, epoch, args, train_log_path,moco_momentum):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mem_losses = AverageMeter('MemLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mem_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if epoch<args.warmup_epochs:
        cur_memory_lr =  args.memory_lr* (epoch+1) / args.warmup_epochs 
    elif args.memory_lr != args.memory_lr_final:
        cur_memory_lr = args.memory_lr_final + 0.5 * \
                   (1. + math.cos(math.pi * (epoch-args.warmup_epochs) / (args.epochs-args.warmup_epochs))) \
                   * (args.memory_lr- args.memory_lr_final)
    else:
        cur_memory_lr = args.memory_lr
    cur_adco_t =args.mem_t
    end = time.time()
    for i, (data0, _, data1, _) in enumerate(train_loader):
        images = [data0, data1]
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
        
        batch_size = images[0].size(0)
        update_sym_network(model, images, args, Memory_Bank, losses, top1, top5, optimizer, criterion, mem_losses,moco_momentum,cur_memory_lr,cur_adco_t)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)
            if args.rank == 0:
                progress.write(train_log_path, i)
                  
    return top1.avg


    

# update query encoder
def update_sym_network(model, images, args, Memory_Bank,
                   losses, top1, top5, optimizer, criterion, mem_losses,
                   moco_momentum,memory_lr,cur_adco_t):
    model.zero_grad()
    
    q_pred, k_pred, q, k = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)

    d_norm1, d1, logits1 = Memory_Bank(q_pred)
    d_norm2, d2, logits2 = Memory_Bank(k_pred)

    # logits: Nx(1+K)
    with torch.no_grad():
        logits_keep1 = logits1.clone()
        logits_keep2 = logits2.clone()
    
    if args.loss == 'caco':
        logits1 /= args.moco_t 
        logits2 /= args.moco_t 

    with torch.no_grad():
        d_norm21, d21, check_logits1 = Memory_Bank(k)
        logits_fix1 = copy.deepcopy(check_logits1)
        check_logits1 = check_logits1.detach()
        filter_index1 = torch.argmax(check_logits1, dim=1)
        labels1 = copy.deepcopy(filter_index1)
        check_logits1 = logits_fix1
        d_norm22, d22, check_logits2 = Memory_Bank(q)
        check_logits2 = check_logits2.detach()
        logits_fix2 = check_logits2
        filter_index2 = torch.argmax(check_logits2, dim=1)
        check_logits2 = logits_fix2
        labels2 = filter_index2

        if args.loss == 'caco_HAPC':
            selected_logitsA = check_logits1[torch.arange(32), labels1]
            A = selected_logitsA.mean()
            tau_0 = args.moco_t
            alphaA = args.alphaA
            A0 = args.A0
            tauA = tau_0 * ( 1 + 1 / alphaA * ( 1 - torch.exp(- (A - A0))))
            logits1 /= tauA
            PA = torch.softmax(logits1, dim=1)[:,0]
            VA = 1. / (1.- PA)
            MACL_lossA = (-VA.detach() * torch.log(PA)).mean()
    

    caco_loss = criterion(logits1, labels1)

    if args.loss=="caco":
        loss = caco_loss
    elif args.loss=='caco_HAPC':
        lamLoss = args.lamLoss
        loss = (1-lamLoss) * caco_loss + MACL_lossA * lamLoss
    else:
        print("请设置loss超参!")
    
    # measure accuracy and record loss
    acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    acc1, acc5 = accuracy(logits2, labels2, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    

    # update memory bank
    with torch.no_grad():
        logits1 = logits_keep1/cur_adco_t#/args.mem_t
        logits2 = logits_keep2/cur_adco_t#/args.mem_t
        p_qd1 = nn.functional.softmax(logits1, dim=1)
        p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]
        g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)
        p_qd2 = nn.functional.softmax(logits2, dim=1)
        p_qd2[torch.arange(logits1.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index2]
        g2 = torch.einsum('cn,nk->ck', [k_pred.T, p_qd2]) / logits2.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2, logits_keep2), dim=0), d_norm2)
        g = -torch.div(g1, torch.norm(d1, dim=0))  - torch.div(g2,torch.norm(d2, dim=0))#/ args.mem_t  # c*k
        g /=cur_adco_t
        
        # g = all_reduce(g) / torch.distributed.get_world_size()
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g #+ args.mem_wd * Memory_Bank.W.data
        # print(Memory_Bank.W, "Befor update: Memory_Bank.W")
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
        # print(Memory_Bank.W, "After update: Memory_Bank.W")
    
    with torch.no_grad():
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        # posi_prob = torch.mean(0.25* (posi_prob + posi_prob2 + posi_prob3 + posi_prob4))
        mem_losses.update(posi_prob.item(), logits.size(0))
    

@torch.no_grad()
def all_reduce(tensor):
    """
    Performs all_reduce(mean) operation on the provided tensors.
    *** Warning ***: torch.distributed.all_reduce has no gradient.
    """
    torch.distributed.all_reduce(tensor, async_op=False)

    return tensor


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class AllReduce(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
    