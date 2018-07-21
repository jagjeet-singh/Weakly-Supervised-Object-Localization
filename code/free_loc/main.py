import pdb
import argparse
import os
import subprocess
import shutil
import time
import sys
import visdom
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# import _init_paths
sys.path.insert(0,'/home/spurushw/reps/hw-wsddn-sol/faster_rcnn')
sys.path.append(os.path.join(os.getcwd(),os.pardir,'faster_rcnn'))
sys.path.append(os.path.join(os.getcwd(),os.pardir))
from logger import Logger
import sklearn
import sklearn.metrics
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# from datasets.factory import get_imdb
from datasets.factory import get_imdb
from custom import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=2, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# parser.add_argument('--pretrained', default=True, type=bool,
#                     help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--log-freq', default=8, type=int,
                    help='Number of batches to log in every epoch')
parser.add_argument('--display-size', default=10, type=int,
                    help='Number of images to display in every batch')
parser.add_argument('--vis',action='store_true')
parser.add_argument('--aws', default=True, type=bool,
                    help='Is AWS being used? - Will be used to fetch dns directly')
parser.add_argument('--visdom-port', default=6006, type=int,
                    help='Port number used for Visdom')
parser.add_argument('--tb-logdir', default='temp')

best_prec1 = 0

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Setting fixed seed which will be used across all the  questions

# def worker_init_fn():


def preprocessHeatMap(hm, cmap = plt.get_cmap('jet'), size=512):
    # pdb.set_trace()
    resize_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((size,size))])
    scaler = MinMaxScaler()
    scaler.fit(hm)
    hm = scaler.transform(hm)
    hm = resize_transform(torch.Tensor(hm).unsqueeze(0))
    hm = np.uint8(cmap(np.array(hm))*255)
    hm = np.transpose(hm, axes=(2,0,1))
    return hm


def get_DNS():
    sub = subprocess.Popen(['curl', '-s' ,'http://169.254.169.254/latest/meta-data/public-hostname'], stdout=subprocess.PIPE)
    dns = sub.stdout.read()
    return dns

def bbFromHeatmap(im, all_heatmaps):
    heatmaps = np.copy(all_heatmaps)
    for c in range(heatmaps.shape[0]):
        hm = heatmaps[c,:,:]
        hm = cv2.resize(hm, (384,384))
        
        # hm_in = cv2.imread('hm2.png')
        # im = cv2.imread('im2.png')
        # hm_in = cv2.resize(hm_in,im.shape[:2])
        t = im.copy()
        t[:,:,0] = im[:,:,2]
        t[:,:,2] = im[:,:,0]
        im = t
        gray_image = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

        # hm = hm>(np.max(hm)*0.99)
        hm = gray_image
        hm = hm>(np.max(hm)*0.9)
        # hm = hm>np.mean(hm)
        row_idx, col_idx = np.where(hm == 1)
        minx = np.min(col_idx)
        miny = np.min(row_idx)
        maxx = np.max(col_idx)
        maxy = np.max(row_idx)
        cv2.rectangle(im, (minx, miny), (maxx, maxy), (0, 204, 0), 2)
        # plt.imshow(im)
        # plt.show()
        cv2.rectangle(hm, (minx, miny), (maxx, maxy), (0, 204, 0), 2)
        # plt.imshow(hm_in)
        # plt.show()
    return im, hm

def main():

    torch.manual_seed(42)
    np.random.seed(42)
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    print(args.pretrained)
    if args.arch=='localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch=='localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)
    # pdb.set_trace()
    model.features = torch.nn.DataParallel(model.features)
    if use_cuda:
        model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer


    # This will find a binary classification loss for each class and then sum over them
    criterion = nn.BCELoss() 
    if use_cuda:
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if use_cuda:
        cudnn.benchmark = True # Helpful if input size doesn't vary every iteration

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    invTrans = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
            std = [ 1., 1., 1. ]),
                               ])

    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # pdb.set_trace()
    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(test_imdb, transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion, invTrans=invTrans, imdb=test_imdb)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()

    #Automatically fetching the instance DNS and passing that to visdom
    address = 'http://'+get_DNS()
    vis = visdom.Visdom(server=address, port=args.visdom_port)

    logger = Logger('hw2_logs', name=args.tb_logdir)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, vis, logger, invTrans, trainval_imdb)

        # evaluate on validation set
        if epoch%args.eval_freq==0 or epoch==args.epochs-1:
            m1, m2 = validate(val_loader, model, criterion, invTrans=invTrans, imdb=test_imdb)

            # Plotting metric1 and metric2
            logger.scalar_summary(tag='validation/metric1', value=m1, step=epoch)
            logger.scalar_summary(tag='validation/metric2', value=m2, step=epoch)
            score = m1*m2
            # remember best prec@1 and save checkpoint
            is_best =  score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    # Training Finished, Doing 1 final validation
    m1, m2 = validate(val_loader, model, criterion, invTrans=invTrans, imdb=test_imdb, final=True, vis=vis)
    print('### Final value for Validation metric1 is {} ###'.format(m1))
    print('### Final value for Validation metric2 is {} ###'.format(m2))


#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, vis, logger,invTrans, imdb):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    # Tentative #
    num_batches =  len(train_loader)
    end = time.time()
    final_loss = 0
    # pdb.set_trace()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        # print('Entered enumerator')
        data_time.update(time.time() - end)

        target = target.type(FloatTensor)
        if use_cuda:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        model_output = model(input_var)

        # Global maxpooling to convert bsx20x11x11 -> bsx20x1x1
        imoutput = F.max_pool2d(model_output, kernel_size=model_output.size()[2:])
        imoutput = imoutput.view(-1,imoutput.size()[1])
        imoutput = F.sigmoid(imoutput)
        loss = criterion(imoutput, target_var)
        # pdb.set_trace()
        final_loss = loss
        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output
        # imoutput = imoutput>0.5

        # measure metrics and record loss

        m1 = metric1(imoutput.data, target_var.data)
        m2 = metric2(imoutput.data, target_var.data)
        # print('Metrics calculated')
        # m1 = metric1(imoutput.data, target)
        # m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))
        
        # TODO: 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        itern_num = epoch*num_batches*args.batch_size+(i+1)*args.batch_size
        global_batch_num = epoch*num_batches+i+1

        # Plotting loss, metric1 and metric2
        logger.scalar_summary(tag='train/loss', value=loss.data[0], step=itern_num)
        logger.scalar_summary(tag='train/metric1', value=m1[0], step=itern_num)
        logger.scalar_summary(tag='train/metric2', value=m2[0], step=itern_num)

        # Plotting histograms
        # if args.arch=='localizer_alexnet':
        logger.model_param_histo_summary(model=model, step=itern_num)
        
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, avg_m1=avg_m1,
                   avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        # pdb.set_trace()
        if (epoch==0 or epoch%5==0) and i in list(range(0,num_batches,num_batches/args.log_freq)):
        # if (epoch==0 or epoch==(args.epochs-1)) and i in list(range(0,num_batches,num_batches/args.log_freq)):
                    
            images = input
            if use_cuda:
                heatmaps = model_output.data.cpu().numpy()
                gt_classes = target.cpu().numpy()
            else:
                heatmaps = model_output.data.numpy()
                gt_classes = target.numpy()

            # Displaying first 4 images of the batch
            for im_num in range(args.display_size):
                image = images[im_num, :, :, :]
                # pdb.set_trace()
                if use_cuda:
                    image = invTrans(image).cpu().numpy()
                else:
                    image = invTrans(image).numpy()
                heatmap = heatmaps[im_num, :, :, :] # For all classes
                gt_class = gt_classes[im_num, :]
                gt_class = np.nonzero(gt_class)[0].astype(int)
                heatmap = heatmap[gt_class, :, :]

                # Plotting images

                # Plotting on visdom
                # if epoch%2==0:
                # if args.arch=='localizer_alexnet':
                vis.image(image, 
                    opts=dict(caption='ep'+str(epoch)+'_'+'iter'+str(itern_num)+'_'+'batch'+str(i)+'_'+'image'+str(im_num),
                        title='ep'+str(epoch)+'_'+'iter'+str(itern_num)+'_'+'batch'+str(i)+'_'+'image'+str(im_num)))
                # vis.images(images, nrow=nrow,opts=dict(caption=str(epoch)+'_'+str(itern_num)+'_'+str(i)+'_'+'image'))
                
                # Plotting on Tensorboard
                image = np.transpose(image, axes=(1,2,0))
                # image = list(image) # since image_summary expects a list of images
                
                logger.image_summary(tag='ep'+str(epoch)+'_'+'iter'+str(itern_num)+'_'+'batch'+str(i)+'_'+'image'+str(im_num), images=[image], step=itern_num)
                # pdb.set_trace()
                #Plotting heatmaps


                for c in range(heatmap.shape[0]):
                    hm = heatmap[c, :, :]
                    # hm = cv2.resize(hm, (512,512))

                    # Plotting on visdom
                    # if epoch%2==0:
                    # pdb.set_trace()
                    hm = preprocessHeatMap(hm)
                    # if args.arch=='localizer_alexnet':
                    vis.image(hm, opts=dict(title='ep'+str(epoch)+'_'+'iter'+str(itern_num)+'_'+'batch'+str(i)+'_'+'image'+str(im_num)+'_'+imdb._classes[gt_class[c]-1]))
                    # vis.heatmap(hm, opts=dict(colormap = 'Jet', title='ep'+str(epoch)+'_'+'iter'+str(itern_num)+'_'+'batch'+str(i)+'_'+'image'+str(im_num)+'_'+imdb._classes[gt_class[c]-1],width=512, height=512))

                    # Normalizing heatmap


                    
                    # vis.image(np.transpose(heatmap,(2,0,1)), opts=dict(title=tag_heat))
                    #vis.heatmap(np.array(np.uint8(heatmap_resize)), opts=dict(colormap='Jet',title=tag_heat))
                    # hmin = np.min(hm)
                    # hmax = np.max(hm)
                    # hm = np.array([(ele - hmin)/(hmax-hmin) for ele in hm])

                    # Plotting on Tensorboard
                    logger.image_summary(tag='ep'+str(epoch)+'_'+'iter'+str(itern_num)+'_'+'batch'+str(i)+'_'+'image'+str(im_num)+'_'+imdb._classes[gt_class[c]-1], images=[hm], step=itern_num)    

def validate(val_loader, model, criterion, invTrans, imdb, final=False, vis=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    # i here is batch_index
    for i, (input, target) in enumerate(val_loader):
        target = target.type(FloatTensor)
        if use_cuda:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        model_output = model(input_var)

        # Global maxpooling to convert bsx20x11x11 -> bsx20x1x1
        imoutput = F.max_pool2d(model_output, kernel_size=model_output.size()[2:])
        imoutput = imoutput.view(-1,imoutput.size()[1])
        imoutput = F.sigmoid(imoutput)
        loss = criterion(imoutput, target_var)
        losses.update(loss.data[0], input.size(0))


        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)

        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   avg_m1=avg_m1, avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals

        #Plotting first image and corresponding heatmap for each batch
        # This part assumes that batch size is 1 for validation. 
        if final and i<40:
            image = input[0,:,:,:]
            if use_cuda:
                heatmaps = model_output.data.cpu().numpy()
                gt_classes = target.cpu().numpy()
            else:
                heatmaps = model_output.data.numpy()
                gt_classes = target.numpy()

            # pdb.set_trace()
            if use_cuda:
                image = invTrans(image).cpu().numpy()
            else:
                image = invTrans(image).numpy()
            heatmap = heatmaps[0, :, :, :] # For all classes
            gt_class = gt_classes[0, :]
            gt_class = np.nonzero(gt_class)[0].astype(int)
            heatmap = heatmap[gt_class, :, :]

            vis.image(image, 
                opts=dict(caption='val'+'_'+'image'+str(i),
                    title='val'+'_'+'image'+str(i)))

            #Plotting bounding boxes
            # bboxImage = bbFromHeatmap(image, heatmap)
            # vis.image(bboxImage, 
            #     opts=dict(caption='val'+'_'+'image'+str(i)+'_bb',
            #         title='val'+'_'+'image'+str(i)+'_bb'))

            for c in range(heatmap.shape[0]):
                hm = heatmap[c, :, :]
                # hm = cv2.resize(hm, (384,384))

                # Plotting on visdom
                hm = preprocessHeatMap(hm, size=384)
                vis.image(hm, opts=dict(title='val'+'_'+'image'+str(i)+'_'+imdb._classes[gt_class[c]-1]))
                # vis.heatmap(hm, opts=dict(title='val'+'_'+'image'+str(i)+'_'+imdb._classes[gt_class[c]-1]))

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'
          .format(avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):

    # return [0]
    nclasses = target.shape[1]
    all_ap = []
    for cid in range(nclasses):
        gt_cls = target[:, cid].cpu().numpy().astype('float32')
        pred_cls = output[:, cid].cpu().numpy().astype('float32')
        if np.count_nonzero(gt_cls) == 0:
            if np.count_nonzero(pred_cls>0.5) == 0:
                ap = 1
            else:
                ap = 0
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        else:
            pred_cls -= 1e-5 * gt_cls
            ap = sklearn.metrics.average_precision_score(
                gt_cls, pred_cls)
        all_ap.append(ap)
    return [np.mean(all_ap)]

def metric2(output, target):
    # return fbeta_score(target, output>0.5, 1, average='weighted')
    nclasses = target.shape[1]
    all_f1 = []
    for cid in range(nclasses):
        gt_cls = target[:, cid].cpu().numpy().astype('float32')
        pred_cls = output[:, cid].cpu().numpy().astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        # pred_cls = pred_cls>0.5
        pred_lbl = pred_cls>0.5
        if np.count_nonzero(gt_cls) == 0:
            # if np.count_nonzero(pred_cls) == 0:
            if np.count_nonzero(pred_lbl) == 0:
                f1 = 1
            else:
                f1 = 0
        else:
            # f1 = sklearn.metrics.recall_score(gt_cls, pred_lbl)
            # f1 = sklearn.metrics.roc_auc_score(gt_cls, pred_cls)
            # f1 = f1_score(gt_cls, pred_lbl)
            f1 = fbeta_score(gt_cls, pred_lbl, 2)
        all_f1.append(f1)
    return [np.mean(all_f1)]
    # nclasses = target.shape[1]
    # all_roc = []
    # for cid in range(nclasses):
    #     gt_cls = target[:, cid].numpy().astype('float32')
    #     pred_cls = output[:, cid].numpy().astype('float32')
    #     roc = sklearn.metrics.roc_auc_score(
    #         gt_cls, pred_cls)
    #     all_roc.append(roc)
    # return np.mean(all_roc)

if __name__ == '__main__':
    main()
