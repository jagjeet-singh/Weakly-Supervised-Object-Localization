from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import subprocess
import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import visdom
from logger import Logger

import cPickle as pkl
import network
from wsddn import WSDDN
from utils.timer import Timer

import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from test import test_net
import pdb
import argparse

try:
    from termcolor import cprint
except ImportError:
    cprint = None

parser = argparse.ArgumentParser(description='PyTorch WSDDN Training')
parser.add_argument('--checkpoint', default=0, type=int, metavar='N',
                    help='Checkpoint (step number) to load')
parser.add_argument('--visdom-port', default=6006, type=int, metavar='N',
                    help='Visdom Port')
parser.add_argument('--end-step', default=50000, type=int, metavar='N',
                    help='End step for training')

args = parser.parse_args()



def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def get_DNS():
    sub = subprocess.Popen(['curl', '-s' ,'http://169.254.169.254/latest/meta-data/public-hostname'], stdout=subprocess.PIPE)
    dns = sub.stdout.read()
    return dns




# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
test_imdb = get_imdb('voc_2007_test')
cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
vis_interval = 2500
train_plot_interval = 500

start_step = 0
end_step = args.end_step
# if args.checkpoint == end_step:
#     end_step+=1
lr_decay_steps = {150000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = True
log_grads = False

remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

torch.manual_seed(42)
np.random.seed(42)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
eval_interval = 1000
hist_interval = 2000
firstLog=1
firstEval = 1


# load imdb and create data later
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.001)

# If checkpoint file exists
ckpt_file = 'models/saved_model/wsddn_' + str(args.checkpoint)+'.h5'
if os.path.exists(ckpt_file):
    network.load_net(ckpt_file, net)
    start_step = args.checkpoint
    print('Loaded from checkpoint:{}'.format(args.checkpoint))
# Using pretrained AlexNet
else:
    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl','r'))
    else:
        pret_net = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net, open('pretrained_alexnet.pkl','wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()
    for name, param in pret_net.items():
        if 'classifier' in name:
            name = str(name[:11]+str(int(name[11])-1)+name[12:])
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue
# pdb.set_trace()
# Move model to GPU and set train mode
net.cuda()
net.train()


# Create optimizer for network parameters
params = list(net.parameters())
optimizer = torch.optim.SGD(params[2:], lr=lr, 
                            momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

#Automatically fetching the instance DNS and passing that to visdom
address = 'http://'+get_DNS()
vis = visdom.Visdom(server=address, port=args.visdom_port)

#p2.xlarge
# vis = visdom.Visdom(server='http://ec2-52-14-238-23.us-east-2.compute.amazonaws.com', port='6006')

logger = Logger('hw2_logs', name='task2_3_5k')

for step in range(start_step, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    #gt_boxes = blobs['gt_boxes']

    # forward
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.data[0]
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps, lr, momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True

    #TODO: evaluate the model every N iterations (N defined in handout)
    if step % eval_interval == 0 and step>0:
        aps = test_net(name='test_weights', net = net, imdb = test_imdb, logger=logger, step=step, visualize=True, thresh = 0.0001)
        if firstEval:
            vis.line(X = np.array([step]), Y = np.array([np.mean(aps)]), win="test/mAP", opts=dict(title='Test mAP'))
            firstEval = 0
        else:
            vis.line(X = np.array([step]), Y = np.array([np.mean(aps)]), win="test/mAP", update="append", opts=dict(title='Test mAP'))
        for i in range(len(aps)):
            logger.scalar_summary(tag='test/AP/'+imdb._classes[i], value=aps[i], step=step)
    #TODO: Perform all visualizations here
    #You can define other interval variable if you want (this is just an
    #example)
    #The intervals for different things are defined in the handout
    if visualize and step%vis_interval==0:
        #TODO: Create required visualizations
        if use_tensorboard:
            print('Logging to Tensorboard')
            logger.scalar_summary(tag='train/loss', value=loss.data[0], step=step)
        if use_visdom:
            print('Logging to visdom')
            if firstLog:
                vis.line(X = np.array([step]), Y = np.array([loss.data[0]]), win="train/loss", opts=dict(title='Train Loss'))
                firstLog=0
            else:
                vis.line(X = np.array([step]), Y = np.array([loss.data[0]]), win="train/loss", update="append", opts=dict(title='Train Loss'))

    if step % hist_interval == 0:
        logger.model_param_histo_summary(model=net, step=step)
    
    # Save model occasionally 
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        save_name = os.path.join(output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX,step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

