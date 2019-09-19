from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Feed') != -1:
        a=1
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)




def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True,offset=False):
    """
    Returns the IoU of two bounding boxes
    """
    if box2.type() is not box1.type():
        box2.type(box1.type())

    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    if offset:
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    else:
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[..., 0]  = prediction[..., 0] - prediction[..., 2] / 2
    box_corner[..., 1]  = prediction[..., 1] - prediction[..., 3] / 2
    box_corner[..., 2]  = prediction[..., 0] + prediction[..., 2] / 2
    box_corner[..., 3]  = prediction[..., 1] + prediction[..., 3] / 2
    prediction[..., :4] = box_corner[..., :4]

    batches,classes,boxes,_ = prediction.shape
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred_class in enumerate(prediction):
        # Filter out confidence scores below threshold
        for c in classes:
            image_pred = image_pred_class[c]
            conf_mask  = (image_pred[:, 4] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            # If none are remaining => process next image
            if not image_pred.size(0):continue
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = image_pred
            # Iterate through all predicted classes
            # Get the detections with the particular class
            detections_class = detections
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output

def non_max_suppression_origin(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output
def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            # grid box就是把原图像和[W,H]的网格对应起来
            # ground truth 的位置通过 grid box 重定向到某一个格点上
            # 所以conf的位置也应该再这个格点上
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            # 3个anchor,表示的是在(W,H)位置上的3个anchor
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            # conf_mask 本来处处为1,指代的是处处不有效
            # mask 是 conf_mask_true 指代的是此位置的conf是有效的
            # conf_mask 把overlap过大的设为0, 之后的操作是conf_mask_false=conf_mask - mask
            # 所以这一步实际上是对3个anchor按照ground_truth进行了排序, 初步筛选出有效的anchor类型
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # 所以这个格点上的3个anchor中大于阈值的都不应该忽略
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # 这个best_n的意思是给出了最好的那个种类的anchor
            # 后面pred_boxes[b, best_n, gj, gi]的意思是,我们选出了每个位置的其中一类(初步筛选出的最好的那一类)anchor,变成了一个W*H*(4pos)的数据
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # 这个是真是的gt_box的位置
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls



def build_targets_5n_An(
    target,
    anchor,
    num_anchors,
    num_classes,
    grid_size,
    pred_conf,
    pred_boxes
    ):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size

    slide_shape=(nB, nC, nA, nG, nG)
    mask = torch.zeros(slide_shape)
    conf_mask = torch.ones(slide_shape)
    tx = torch.zeros(slide_shape)
    ty = torch.zeros(slide_shape)
    tw = torch.zeros(slide_shape)
    th = torch.zeros(slide_shape)
    tconf = torch.ByteTensor(nB, nC, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nC, nA, nG, nG).fill_(0)


    anchor_awh = anchor.reshape(nG,nG,nA,4).permute(2,0,1,3)
    nGT = 0
    nCorrect = 0
    response_classes=[]
    for b in range(nB):
        response_class=set()
        for t in range(target.shape[1]):
            if target[b,t].sum() == 0:continue
            nGT += 1
            # Convert to position relative to box
            target_now=target[b,t].type(anchor.type())
            c=int(target_now[0])
            response_class.add(c)
            target_box= target_now[1:].unsqueeze(0)
            iou       = bbox_iou(anchor,target_box,False)
            iou_awh   = iou.reshape(nG,nG,nA).permute(2,0,1)
            mask_awh  = iou_awh>0.5
            # if positive case less than 20, use the largest 20 response
            min_pos=20
            if torch.sum(mask_awh)<min_pos:
                mask_vec=torch.zeros_like(iou)
                mask_vec[np.argsort(iou)[-min_pos:]]=1
                mask_awh=mask_vec.reshape(nG,nG,nA).permute(2,0,1)*torch.sign(iou_awh)
                mask_awh=mask_awh.type(torch.ByteTensor)
            # Masks
            mask[b, c, :] = mask_awh
            positive_anchor = anchor_awh[mask_awh]
            tx[b, c, mask_awh] = positive_anchor[:,0]-target_box[:,0]
            ty[b, c, mask_awh] = positive_anchor[:,1]-target_box[:,1]
            tw[b, c, mask_awh] = (target_box[:,2] / positive_anchor[:,2] + 1e-16).log()
            th[b, c, mask_awh] = (target_box[:,3] / positive_anchor[:,3] + 1e-16).log()
            # One-hot encoding of label
            tconf[b, c, mask_awh] = 1 #useless for this case
            pred_good_box = pred_conf[b, c]>0.5
            pred_box_iou  = bbox_iou(pred_boxes[b,c][pred_good_box],target_box, x1y1x2y2=False)
            if len(pred_box_iou)>0:nCorrect += torch.sum(pred_box_iou>0.5)/len(pred_box_iou)
        response_classes.append(response_class)
    return nGT,nCorrect,mask, conf_mask, tx, ty, tw, th, tconf,response_classes
def build_targets_5n(
    target,
    anchors,
    num_anchors,
    num_classes,
    grid_size,
    ignore_thres,
    img_dim,
    pred_conf,
    pred_boxes
    ):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    slide_shape=(nB, nC, nA, nG, nG)
    mask = torch.zeros(slide_shape)
    conf_mask = torch.ones(slide_shape)
    tx = torch.zeros(slide_shape)
    ty = torch.zeros(slide_shape)
    tw = torch.zeros(slide_shape)
    th = torch.zeros(slide_shape)
    tconf = torch.ByteTensor(nB, nC, nA, nG, nG).fill_(0)
    tcls  = torch.ByteTensor(nB, nC, nA, nG, nG).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b,t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            c,gx,gy,gw,gh=target[b,t]
            c  =int(c)
            gx *= nG
            gy *= nG
            gw *= nG
            gh *= nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b,c, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)

            # Masks
            mask[b, c, best_n, gj, gi] = 1
            conf_mask[b, c, best_n, gj, gi] = 1
            # Coordinates
            tx[b, c, best_n, gj, gi] = gx - gi
            ty[b, c, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, c, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, c, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label,only the best is chosen
            tconf[b, c, best_n, gj, gi] = 1

            gt_box   = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            pred_box = pred_boxes[b, c, best_n, gj, gi].unsqueeze(0)
            score = pred_conf[b, c, best_n, gj, gi]
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            #实际上是正确答案是否在预测集中的准确率,严格来说应该是覆盖率
            if iou > 0.5 and score > 0.5:nCorrect += 1

    return nGT,nCorrect,mask, conf_mask, tx, ty, tw, th, tconf

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])
