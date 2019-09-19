from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets,build_targets_5n,build_targets_5n_An
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from anchor_generator import Anchor

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()


        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            #filters = sum([output_filters[layer_i] for layer_i in layers])
            filters = 0
            for layer_i in layers:
                    if(layer_i > 0):
                        filters += output_filters[layer_i+1]
                    else:
                        filters += output_filters[layer_i]
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)

        elif module_def["type"] == "feed_conv2d":
            filters = int(module_def["anchors_num"])*5
            if "out_channel" in module_def:filters=int(module_def["out_channel"])
            modules.add_module(
                "feed_conv_%d" % i,
                FeedConv2d(
                    in_channels    =output_filters[-1],
                    out_channel_unit=filters,
                    kernel_size    =int(module_def["size"]),
                    stride       =int(module_def["stride"])
                ),
            )

        elif module_def["type"] == "fyolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = FYOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)

        elif module_def["type"] == "myolo":
            ratios=[float(x) for x in module_def["ratios"].split(",")]
            scales=[float(x) for x in module_def["scales"].split(",")]
            #ratios=[0.33, 1, 3]
            #scales=[1]
            num_anchors_should=output_filters[-1]/5
            num_anchors=len(ratios)*len(scales)
            assert num_anchors_should == num_anchors
            anchor_generator=Anchor(ratios,scales)
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = MYOLOLayer(anchor_generator,num_anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BinaryFocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, alpha=None, gamma=2, size_average=True):
        super().__init__()
        class_num=2
        if alpha is None:
            self.alpha =1
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, pos_poss, OneHot,class_pick=None):
        alpha=self.alpha
        if class_pick is not None:
            temp=torch.stack([pos_poss[i,c] for i,c in enumerate(class_pick)])
            pos_poss=temp
            temp=torch.stack([OneHot[i,c] for i,c in enumerate(class_pick)])
            OneHot=temp
        probs = pos_poss*OneHot+(1-pos_poss)*(1-OneHot)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def batch_class_pick(tensor,class_pick):
    return torch.stack([tensor[i,c] for i,c in enumerate(class_pick)])

class FeedConv2d(nn.Module):

    def __init__(self,in_channels,out_channel_unit,kernel_size=(1,1),stride=1,padding=0):
        super(FeedConv2d, self).__init__()
        self.in_channels=in_channels
        self.unit_out_c_num =out_channel_unit
        self.stride  = stride
        self.padding = padding
        self.kernel_size=kernel_size
    def forward(self,_input):
        x,kernels,biases=_input
        batch,in_channel,W,H=x.shape
        assert in_channel==self.in_channels
        if isinstance(kernels,list):
            for kernel in kernels:
                out_channel,in_channel,W,H=kernel.shape
                assert in_channel==self.in_channels
                assert out_channel==self.unit_out_c_num
                assert W == self.kernel_size
                assert H == self.kernel_size
            full_kernel=torch.cat(kernels)
        else:
            full_kernel=kernels
        if isinstance(biases,list):
            for bias in biases:
                out_channel=bias.shape[0]
                assert out_channel==self.unit_out_c_num
            full_bias  =torch.cat(biases)
        else:
            full_bias  =biases
        return F.conv2d(x,full_kernel,bias=full_bias,stride=self.stride,padding=self.padding)

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor  = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls  = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true  = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false])\
                      + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output

class FYOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(FYOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor  = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB,-1,nA, nG, nG).permute(0, 2, 3, 4, 1).contiguous()
        left_channel = prediction.shape[-1]-5


        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls  = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h


        if targets is not None:
            raise NotImplementedError

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, left_channel),
                ),
                -1,
            )
            return output

def random_choice(set_big,set_small,num):
    set_big  =set(set_big)
    set_small=set(set_small)
    set_dif  =list(set_big-set_small)
    num_other_pick=num-len(set_small)
    if num_other_pick >=0:
        set_other_pick=np.random.choice(set_dif,num_other_pick)
        set_out=list(set_other_pick)+list(set_small)
    else:
        set_out=list(set_small)[:num]
    return set_out

class MYOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchor_generator,num_anchors, num_classes, img_dim):
        super(MYOLOLayer, self).__init__()
        self.anchor_generator=anchor_generator
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 #4 coor +1 conf
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        self.ce_loss  = nn.CrossEntropyLoss()  # Class loss
        self.bfl_loss = BinaryFocalLoss()

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nT = self.bbox_attrs
        # x is [batch,anchor*5*class,W,H]
        # x is [1,3*5*80,5,5]
        # x is [batch,W,H,anchor*5,class]
        nB = x.size(0)
        nG = x.size(2)
        nC = x.size(1)//(nT*nA)

        anchor=self.anchor_generator.get_anchor(nA,nG,nG)
        anchor_awh=self.anchor_generator.get_anchor_awh(nA,nG,nG)

        anchor   =anchor.cuda() if x.is_cuda else anchor
        anchor_awh=anchor_awh.cuda() if x.is_cuda else anchor_awh
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor  if x.is_cuda else torch.LongTensor
        ByteTensor  = torch.cuda.ByteTensor  if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nC  ,nA , nT, nG, nG).permute(0, 1, 2, 4, 5, 3).contiguous()
        #                batch,class, anchor, 5        , W , H
        # target shape   batch,class, anchor,4 pos+1 pred, W,H
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf


        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.detach() + anchor_awh[..., 0]
        pred_boxes[..., 1] = y.detach() + anchor_awh[..., 1]
        pred_boxes[..., 2] = torch.exp(w.detach()) * anchor_awh[..., 2]
        pred_boxes[..., 3] = torch.exp(h.detach()) * anchor_awh[..., 3]

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()
                self.bfl_loss = self.bfl_loss.cuda()
            nGT, nCorrect,mask, conf_mask, tx, ty, tw, th, tconf,rpc= build_targets_5n_An(
                    target    = targets.detach().cpu(),
                    anchor    = anchor.detach().cpu(),
                    num_anchors= nA,
                    num_classes= self.num_classes,
                    grid_size  = nG,
                    pred_boxes =pred_boxes.detach().cpu(),
                    pred_conf  =pred_conf.detach().cpu()
            )

            #nProposals  = int((pred_conf > 0.5).sum().item())
            recall     = float(nCorrect / nGT) if nGT else 1
            precision  = 0
            # Handle masks
            mask      = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))
            # mask.shape=(nB, nC, nA, nG, nG)
            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true  = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x    = self.mse_loss(x[mask], tx[mask])
            loss_y    = self.mse_loss(y[mask], ty[mask])
            loss_w    = self.mse_loss(w[mask], tw[mask])
            loss_h    = self.mse_loss(h[mask], th[mask])
            class_choice=torch.LongTensor([random_choice(range(nC),c,5) for c in rpc])
            #loss_conf = self.bfl_loss(pred_conf, tconf)
            pred_conf=batch_class_pick(pred_conf,class_choice)
            tconf    =batch_class_pick(tconf,class_choice)
            loss_conf = self.bce_loss(pred_conf,tconf)
            #loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            #  purpose of cross-entropy is want to count the influence between different target
            #  bet here we dont's want different target object entangle with each other
            loss_cls = 0
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                recall,
                precision
                )

        else:
            # If not in training phase return predictions
            output = torch.cat(
            (
            pred_boxes.view(nB,nC, -1, 4),
            pred_conf.view(nB,nC, -1, 1)
            ),
            -1,
            )
            return output

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]


    def get_layer_output(self,x,index):
        assert type(index) is int or list
        if type(index) is int:
            return self.get_layer_sin_output(x,index)
        if type(index) is list:
            return self.get_layer_mul_output(x,index)


    def get_layer_mul_output(self,x,indexes):
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module(x)
            layer_outputs.append(x)
        for idx in indexes:
            out=layer_outputs[idx]
            output.append(out)
        return output

    def get_layer_sin_output(self,x,index):
        output = []
        length = len(self.module_list)
        if index >= length:
            print("the index should be in 0-{}".format(length))
            return None
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module(x)
            layer_outputs.append(x)
            if i==index:
                return x
        print("the index should be in {}")

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

class DarkSiamRPN(nn.Module):
    def __init__(self, SiamBackBone, feature_out=256, anchor=5):
        feat_in = 256
        #set_trainable(SiamBackBone,mode='fixed all'):
        self.feature_out = feature_out
        self.anchor      = anchor
        super(DarkSiamRPN, self).__init__()
        self.backbone    = SiamBackBone
        self.conv_r1     = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2     = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1   = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2   = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []

        self.cfg = {}

    def featureExtract(self,x):
        with torch.no_grad():
            fea_map = self.backbone.get_layer_output(x,91)
        return fea_map

    def forward(self, x):
        x_f = self.featureExtract(x)
        r2_input  =self.conv_r2(x_f)
        cls2_input=self.conv_cls2(x_f)
        Similiarity= self.batch_conv2d(r2_input,   self.r1_kernel)
        cout       = self.batch_conv2d(cls2_input, self.cls1_kernel)
        rout       = self.regress_adjust(Similiarity)
        rout       = rout.permute(1, 2, 3, 0).contiguous().view(4, -1)
        cout       = cout.permute(1, 2, 3, 0).contiguous().view(2, -1)
        return rout, cout

    def temple(self, z):
        z_f = self.featureExtract(z)
        r1_kernel_raw   = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size     = r1_kernel_raw.data.size()[-1]
        self.r1_kernel   = r1_kernel_raw.view(-1,self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(-1,self.anchor*2, self.feature_out, kernel_size, kernel_size)

    def train_phase(self,templete,detection):
        self.temple(templete)
        return self.forward(detection)

    def batch_conv2d(self,b_input,b_kernel):
        batch=b_input.data.size()[0]
        outs=[]
        for i in range(batch):
            conv_input =b_input[i:i+1]
            conv_kernel=b_kernel[i]
            out=F.conv2d(conv_input,conv_kernel)
            outs.append(out)
        return torch.cat(outs)

    def trainable_para(self):
        para_name_list=[]
        for name,p in self.named_parameters():
            if p.requires_grad:
                para_name_list.append(name)
        return para_name_list
    def save_state_dict(self):
        trainable_para_list=self.trainable_para()
        save_dict=self.state_dict().copy()
        save_dict.clear()
        for name in trainable_para_list:
            save_dict[name]=self.state_dict()[name]
        return save_dict

class DarkSiamTracking(nn.Module):
        def __init__(self,  feature_generate,
                            kernel_generate,
                            anchor=5):
            super(DarkSiamRPN, self).__init__()
            feat_in = 256
            feature_out=512
            #set_trainable(SiamBackBone,mode='fixed all'):
            self.feature_out = feature_out
            self.anchor      = anchor_num

            self.backbone    = feature_generate
            self.kernel_g    = kernel_generate

            self.conv_kernel = nn.Conv2d(feat_in, feature_out*6*anchor,1)


            # self.conv_r2     = nn.Conv2d(feat_in, feature_out, 3)
            # self.conv_cls2   = nn.Conv2d(feat_in, feature_out, 3)

            self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

            self.r1_kernel = []
            self.cls1_kernel = []

            self.cfg = {}

        def kernelExtract(self,y):
            kernel = self.kernel_g.get_layer_output(x,91)#[1,256,10,10]
            kernel = self.conv_kernel(kernel)#[1,512*anchor*6,10,10]
            return kernel

        def featureExtract(self,x):
            with torch.no_grad():
                fea_map = self.backbone.get_layer_output(x,92)#[1,512,26,26]
            #FPN-->conv3*3-->yolo[92]--conv2d-->[1,255,w,h] normal yolo result
            #fea_map = self.backbone.get_layer_output(x,91)
            return fea_map

        def forward(self, x):
            x_f = self.featureExtract(x)
            # r2_input   = self.conv_r2(x_f)
            # cls2_input = self.conv_cls2(x_f)
            cout_rout  = self.batch_conv2d(x_f, self.kernel)#[1,512,26,26]
            cout_rout  = rout.permute(1, 2, 3, 0).contiguous().view(6, -1)
            cout       = cout_rout[:2,:]
            rout       = cout_rout[2:,]
            return rout, cout

        def temple(self, z):
            z_f              = self.kernelExtract(z)
            kernel_size      = z_f.data.size()[-1]
            self.kernel      = z_f.view(-1,self.anchor*6, self.feature_out, kernel_size, kernel_size)

        def train_phase(self,templete,detection):
            self.temple(templete)
            return self.forward(detection)

        def batch_conv2d(self,b_input,b_kernel):
            batch=b_input.data.size()[0]
            outs=[]
            for i in range(batch):
                conv_input =b_input[i:i+1]
                conv_kernel=b_kernel[i]
                out=F.conv2d(conv_input,conv_kernel)
                outs.append(out)
            return torch.cat(outs)

        def trainable_para(self):
            para_name_list=[]
            for name,p in self.named_parameters():
                if p.requires_grad:
                    para_name_list.append(name)
            return para_name_list
        def save_state_dict(self):
            trainable_para_list=self.trainable_para()
            save_dict=self.state_dict().copy()
            save_dict.clear()
            for name in trainable_para_list:
                save_dict[name]=self.state_dict()[name]
            return save_dict

class MDarknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(MDarknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "recall", "precision"]


    def get_layer_output(self,x,index):
        assert type(index) is int or list
        if type(index) is int:
            return self.get_layer_sin_output(x,index)
        if type(index) is list:
            return self.get_layer_mul_output(x,index)

    def forward(self,_input ,targets=None):
        if len(_input)==3:
            x, feed_cls_k_b,index =_input
        else:
            x, feed_cls_k_b =_input
            index=None
        assert type(index) is int or list or NoneType
        if type(index) is int:index=[index]
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        pick_outputs  = []
        level=0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "feed_conv2d":
                kernels,biases=feed_cls_k_b[level]
                level+=1
                x = module([x,kernels,biases])
            elif module_def["type"] == "myolo" or 'fyolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    #print(losses)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            if index is not None:
                if len(index) == 0:return pick_outputs
                if i in index:
                    pick_outputs.append(x)
                    _=index.remove(i)
            layer_outputs.append(x)
        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if is_training else output

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
