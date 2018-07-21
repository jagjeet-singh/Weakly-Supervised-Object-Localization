# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
import pdb
import numpy as np
import numpy.random as npr
import cv2
import os
import torchvision.datasets
import torchvision.datasets.folder as dsetfolder
import torchvision.transforms as transforms


# >>>> obsolete, because it depends on sth outside of this project
from fast_rcnn.config import cfg
# <<<< obsolete
from utils.blob import prep_im_for_blob, im_list_to_blob


def get_weak_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}
    
    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    boxscores_blob = np.zeros(0, dtype=np.float32)
    labels_blob = np.zeros((0,num_classes), dtype=np.float32)
    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    thislabels = np.zeros((1,num_classes))
    det_prior_blob = np.zeros((0, num_classes), dtype=np.float32)
    # all_overlaps = []


    for im_i in xrange(num_images):
        labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
            = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                           num_classes)
        #TODO: same as get_minibatch, but we only use the image-level labels
        #So blobs['labels'] should contain a 1x20 binary vector for each image 
        # Add to RoIs blob                                                  
        rois = _project_im_rois(im_rois, im_scales[im_i])                   
        batch_ind = im_i * np.ones((rois.shape[0], 1))                      
        rois_blob_this_image = np.hstack((batch_ind, rois))                 
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels = roidb[im_i]['gt_vec'].reshape(1, num_classes) 
        labels_blob = np.vstack((labels_blob, labels)) 
        
    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob
    blobs['im_name'] = os.path.basename(roidb[0]['image'])
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    return blobs


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['gt_ishard'] = roidb[0]['gt_ishard'][gt_inds]  \
            if 'gt_ishard' in roidb[0] else np.zeros(gt_inds.size, dtype=int)
        # blobs['gt_ishard'] = roidb[0]['gt_ishard'][gt_inds]
        blobs['dontcare_areas'] = roidb[0]['dontcare_areas'] * im_scales[0] \
            if 'dontcare_areas' in roidb[0] else np.zeros([0, 4], dtype=float)
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
        blobs['im_name'] = os.path.basename(roidb[0]['image'])

    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs

            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob
        blobs['im_name'] = os.path.basename(roidb[0]['image'])
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    #labels = roidb['labels']
    labels = None
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.where(roidb['gt_classes']==0)[0]

    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = None, None
    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    mean=np.array([[[0.485, 0.456, 0.406]]])
    std=np.array([[[0.229, 0.224, 0.225]]])
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])/255.0
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        #im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
        #                                cfg.TRAIN.MAX_SIZE)
        im, im_scale = prep_im_for_blob(im, target_size,
                                        cfg.TRAIN.MAX_SIZE,
                                       mean=mean,
                                       std=std)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
