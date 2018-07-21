import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from utils.blob import im_list_to_blob, prep_im_for_blob
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
from roi_pooling.modules.roi_pool import RoIPool
# from vgg16 import VGG16

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self, classes=None, debug=False, training=True):
        super(WSDDN, self).__init__()
        self.training = training
        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.roi_pool = RoIPool(6, 6, 1.0/16)
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            )
        self.score_cls = nn.Linear(4096, 20)
        self.score_det = nn.Linear(4096, 20)
    
        # loss
        self.cross_entropy = None

        # for log
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy
	
    def forward(self, im_data, rois, im_info, gt_vec=None,
                gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        rois = network.np_to_variable(rois, is_cuda=True)
	
        #TODO: Use im_data and rois as input
        # compute cls_prob which are N_roi X 20 scores
        # Checkout faster_rcnn.py for inspiration

        features = self.features(im_data)
        pooled_features = self.roi_pool(features, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.classifier(x)
        # shape of x should be num_roi x 4096
        cls_score = self.score_cls(x)
        cls_softmax = F.softmax(cls_score, dim=1)
        det_score = self.score_det(x)
        det_softmax = F.softmax(det_score, dim=0)
        final_score = cls_softmax*det_softmax
        cls_prob = torch.sum(final_score, 0).view(-1, self.n_classes) # 1xc

        if self.training:
            label_vec = network.np_to_variable(gt_vec, is_cuda=True)
            label_vec = label_vec.view(-1, self.n_classes)
            self.cross_entropy = self.build_loss(torch.clamp(cls_prob, min=0.0, max=1.0), label_vec)
            # self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return final_score
    
    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector 
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called

        return F.binary_cross_entropy(cls_prob, label_vec, size_average=False)

	return loss

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        im_orig = im.astype(np.float32, copy=True)/255.0
        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []
        mean=np.array([[[0.485, 0.456, 0.406]]])
        std=np.array([[[0.229, 0.224, 0.225]]])
        for target_size in self.SCALES:
            im, im_scale = prep_im_for_blob(im_orig, target_size,
                                            self.MAX_SIZE,
                                            mean=mean,
                                            std=std)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def load_from_npz(self, params):
        self.features.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 
                 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)

