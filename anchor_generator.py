import numpy as np
import torch
class Anchor(object):
    def __init__(self,ratios,scales):
        self.eps  = 0.01
        self.ratios=ratios
        self.scales=scales
        self.anchor_num=len(ratios)*len(scales)
        self.anchor=None
        self.anchor_awh=None

    def get_anchor(self,nA,nW,nH):
        '''
        return the corner format anchor
        '''
        anchor_num=self.anchor_num
        assert anchor_num==nA
        if self.anchor is None:
            anchor=self.generate_anchor_as_ratios(nW,nH,self.ratios,self.scales)
            #self.unclip_an_center=anchor
            #anchor=self.center_to_corner(anchor)
            #anchor=self.ratio_clip(anchor)
            self.anchor = anchor
        return self.anchor
    def get_anchor_awh(self,nA,nW,nH):
        anchor_num=self.anchor_num
        assert anchor_num==nA
        if self.anchor_awh is None:
            anchor=self.get_anchor(nA,nW,nH)
            self.anchor_awh=anchor.reshape(nW,nH,anchor_num,4).permute(2,0,1,3)
        return self.anchor_awh
    def generate_anchor_as_ratios(self,aw,ah,ratios,scales):
        #aw=ah=50
        #ratios      = [0.33, 0.5, 1, 2, 3]
        #scales      = [0.05]
        avg_stride  = 1.0/np.sqrt(aw*ah)
        anchor_num  = len(ratios) * len(scales)
        anchor      = np.zeros((anchor_num, 4),  dtype=np.float32)
        count       = 0
        for ratio in ratios:
            ws = np.sqrt(1 / ratio)
            hs = ws * ratio
            for scale in scales:
                wws = ws * scale
                hhs = hs * scale
                anchor[count, :] = [0,0,wws,hhs]
                count += 1
        # pad_x=np.max(anchor[:,2])/2
        # pad_y=np.max(anchor[:,3])/2
        x = np.arange(aw)/(aw-1)
        y = np.arange(ah)/(ah-1)
        # x = (x+pad_x)/(1+2*pad_x)
        # y = (y+pad_y)/(1+2*pad_y)
        # anchor[:, 2]/=(1+2*pad_x)
        # anchor[:, 3]/=(1+2*pad_y)
        anchor = np.tile(anchor, (aw * ah,1))

        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.flatten(), yy.flatten()  # coordinate of x and y
        xx, yy = np.tile(np.stack([xx,yy],axis=1),anchor_num).reshape(-1,2).transpose()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return torch.Tensor(anchor)
    # float
    def center_to_corner(self,box):
        if 'torch' in str(type(box)):
            box_=box.new(box.shape)
        elif 'numpy' in str(type(box)):
            box = box.copy()
            box_ = np.zeros_like(box, dtype = np.float32)
        box_[...,0]=box[...,0]-box[...,2]/2
        box_[...,1]=box[...,1]-box[...,3]/2
        box_[...,2]=box[...,0]+box[...,2]/2
        box_[...,3]=box[...,1]+box[...,3]/2
        return box_

    def ratio_clip(self,box):
        # corner box
        # if the box is the ratio format
        # min,max box range is [0,1]
        box[box>1]=1
        box[box<0]=0
        return box
    # float
    def corner_to_center(self,box):
        if 'torch' in str(type(box)):
            box_=box.new(box.shape)
        elif 'numpy' in str(type(box)):
            box = box.copy()
            box_ = np.zeros_like(box, dtype = np.float32)
        box_[...,0]=(box[...,2]+box[...,0])/2
        box_[...,1]=(box[...,3]+box[...,1])/2
        box_[...,2]=(box[...,2]-box[...,0])
        box_[...,3]=(box[...,3]-box[...,1])
        return box_

    def iou_numpy(self,box1,box2):
        box1, box2 = box1.copy(), box2.copy()
        N=box1.shape[0]
        K=box2.shape[0]
        box1=np.array(box1.reshape((N,1,4)))+np.zeros((1,K,4))#box1=[N,K,4]
        box2=np.array(box2.reshape((1,K,4)))+np.zeros((N,1,4))#box1=[N,K,4]
        x_max=np.max(np.stack((box1[:,:,0],box2[:,:,0]),axis=-1),axis=2)
        x_min=np.min(np.stack((box1[:,:,2],box2[:,:,2]),axis=-1),axis=2)
        y_max=np.max(np.stack((box1[:,:,1],box2[:,:,1]),axis=-1),axis=2)
        y_min=np.min(np.stack((box1[:,:,3],box2[:,:,3]),axis=-1),axis=2)
        tb=x_min-x_max
        lr=y_min-y_max
        tb[np.where(tb<0)]=0
        lr[np.where(lr<0)]=0
        over_square=tb*lr
        all_square=(box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])+(box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])-over_square
        return over_square/all_square

    def bbox_iou(self,box1, box2, x1y1x2y2=True,offset=False):
        """
        Returns the IoU of two bounding boxes
        box1 is for anchors
        box2 is for groundtruth
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

    def build_targets(self,pred_boxes, pred_cls, target):

        ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # Convert to position relative to box
        # Convert to position relative to box
        target_boxes_center= target[:, 2:]
        anchor_center = self.get_anchor(nA,nG,nG)
        # Get anchors with best iou
        ious = torch.stack([self.bbox_iou(anchor_center, target_center.unsqueeze(0),False) for target_center in target_boxes_center],1)
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = target[:, :2].long().t()
        pA=best_n%5
        aX,aY,aW,aH=anchor_center[best_n].t()
        gX,gY,gW,gH=target_boxes_center.t()
        pX=(aX*(nG-1)).long()
        pY=(aY*(nG-1)).long()

        # Set masks
        obj_mask[b, pA, pX,pY]   = 1
        noobj_mask[b, pA, pX,pY] = 0
        # Set noobj mask to zero where iou exceeds ignore threshold
        #noobj_mask[b][iou_awh > 0.5] = 0
        # Coordinates
        tx[b, pA, pX,pY] = gX - aX
        ty[b, pA, pX,pY] = gY - aY
        # Width and height
        tw[b, pA, pX,pY] = torch.log(gW / aW + 1e-16)
        th[b, pA, pX,pY] = torch.log(gH / aH + 1e-16)
        # One-hot encoding of label
        tcls[b, pA, pX,pY, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, pA, pX,pY] = (pred_cls[b, pA, pX,pY].argmax(-1) == target_labels).float()
        iou_scores[b, pA, pX,pY] = self.bbox_iou(pred_boxes[b, pA, pX,pY], target_boxes_center, x1y1x2y2=False)
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    def plot_anchor(self,an,gt,size,number):
        import cv2
        abs_an = an*size
        abs_gt = gt*size
        gt_corner=self.center_to_corner(abs_gt)
        an_corner=self.center_to_corner(abs_an)
        backgroud=np.zeros((size,size,3))
        num=0
        for x1,y1,x2,y2 in an_corner.astype('int'):
            _=cv2.rectangle(backgroud,(x1,y1),(x2,y2), (0, 255, 255), 1)
            if num>number:break
            num+=1
        for x1,y1,x2,y2 in gt_corner.astype('int'):
            _=cv2.rectangle(backgroud,(x1,y1),(x2,y2), (255, 255, 255), 1)
        return backgroud

class Anchor_ms(object):
    """
    stable version for anchor generator
    """
    def __init__(self, score_size   ,
                        ratios       ,
                        scales       ,
                        total_stride ,
                        threshold_pos,
                        threshold_neg):
        self.score_size   = score_size
        self.ratios       = ratios
        self.scales       = scales
        self.total_stride = total_stride
        self.threshold_pos= threshold_pos
        self.threshold_neg= threshold_neg

        anchors        =self.generate_anchor()
        self.anchors   = anchors.copy()
        self.rel_anchor_corner=np.stack([anchors[:,0]-anchors[:,2]/2,anchors[:,1]-anchors[:,3]/2,anchors[:,0]+anchors[:,2]/2,anchors[:,1]+anchors[:,3]/2]).transpose()
        self.eps       = 0.01


    def generate_anchor(self):
        total_stride= self.total_stride
        ratios      = self.ratios
        score_size  = self.score_size
        scales      = self.scales
        anchor_num  = len(ratios) * len(scales)
        anchor      = np.zeros((anchor_num, 4),  dtype=np.float32)
        size        = total_stride * total_stride
        count       = 0
        for ratio in ratios:
            # ws = int(np.sqrt(size * 1.0 / ratio))
            ws = int(np.sqrt(size / ratio))
            hs = int(ws * ratio)
            for scale in scales:
                wws = ws * scale
                hhs = hs * scale
                anchor[count, 0] = 0
                anchor[count, 1] = 0
                anchor[count, 2] = wws
                anchor[count, 3] = hhs
                count += 1

        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size / 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        xx -= np.average(xx)
        yy -= np.average(yy)
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

        return anchor

    # float
    def diff_anchor_gt(self, gt):
        eps = self.eps
        t_w,t_h   = gt
        gt_corner =np.array([0,0,t_w,t_h])
        anchors, gt = self.anchors.copy(), gt_corner.copy()
        diff = np.zeros_like(anchors, dtype = np.float32)
        diff[:,0] = (gt[0] - anchors[:,0])/(anchors[:,2] + eps)
        diff[:,1] = (gt[1] - anchors[:,1])/(anchors[:,3] + eps)
        diff[:,2] = np.log((gt[2] + eps)/(anchors[:,2] + eps))
        diff[:,3] = np.log((gt[3] + eps)/(anchors[:,3] + eps))
        return diff

    # float
    def center_to_corner(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype = np.float32)
        box_[...,0]=box[...,0]-(box[:,2]-1)/2
        box_[...,1]=box[...,1]-(box[:,3]-1)/2
        box_[...,2]=box[...,0]+(box[:,2]-1)/2
        box_[...,3]=box[...,1]+(box[:,3]-1)/2
        box_ = box_.astype(np.float32)
        return box_

    # float
    def corner_to_center(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype = np.float32)
        box_[...,0]=box[...,0]+(box[...,2]-box[...,0])/2
        box_[...,1]=box[...,1]+(box[...,3]-box[...,1])/2
        box_[...,2]=box[...,2]-box[...,0]
        box_[...,3]=box[...,3]-box[...,1]
        box_ = box_.astype(np.float32)
        return box_

    def pos_neg_anchor(self, gt, pos_num, neg_num):
        assert len(gt)==2
        #now only for center relative gt
        t_w,t_h   = gt
        gt_corner = np.array([-t_w/2,-t_h/2,t_w/2,t_h/2]).reshape(1, 4)
        threshold_pos=self.threshold_pos
        threshold_neg=self.threshold_neg
        an_corner = self.rel_anchor_corner
        iou_value = self.iou(an_corner, gt_corner).reshape(-1) #(1445)
        max_iou   = max(iou_value)
        pos, neg  = np.zeros_like(iou_value, dtype=np.int32), np.zeros_like(iou_value, dtype=np.int32)

        # pos
        pos_cand  = np.argsort(iou_value)[::-1][:30]
        pos_index = np.random.choice(pos_cand, pos_num, replace = False)
        if max_iou > threshold_pos:
            pos[pos_index] = 1

        # neg
        neg_cand = np.where(iou_value < threshold_neg)[0]
        neg_ind  = np.random.choice(neg_cand, neg_num, replace = False)
        neg[neg_ind] = 1

        return pos, neg

    def iou_numpy(self,box1,box2):
        box1, box2 = box1.copy(), box2.copy()
        N=box1.shape[0]
        K=box2.shape[0]
        box1=np.array(box1.reshape((N,1,4)))+np.zeros((1,K,4))#box1=[N,K,4]
        box2=np.array(box2.reshape((1,K,4)))+np.zeros((N,1,4))#box1=[N,K,4]
        x_max=np.max(np.stack((box1[:,:,0],box2[:,:,0]),axis=-1),axis=2)
        x_min=np.min(np.stack((box1[:,:,2],box2[:,:,2]),axis=-1),axis=2)
        y_max=np.max(np.stack((box1[:,:,1],box2[:,:,1]),axis=-1),axis=2)
        y_min=np.min(np.stack((box1[:,:,3],box2[:,:,3]),axis=-1),axis=2)
        tb=x_min-x_max
        lr=y_min-y_max
        tb[np.where(tb<0)]=0
        lr[np.where(lr<0)]=0
        over_square=tb*lr
        all_square=(box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])+(box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])-over_square
        return over_square/all_square

    def bbox_iou(self,box1, box2, x1y1x2y2=True,offset=False):
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

    def get_pos_neg_diff(self,gt_box_in_detection, pos_num=16, neg_num=48):
        pos, neg       = self.pos_neg_anchor(gt_box_in_detection,pos_num,neg_num)
        diff           = self.diff_anchor_gt(gt_box_in_detection)
        pos, neg, diff = pos.reshape((-1, 1)), neg.reshape((-1,1)), diff.reshape((-1, 4))
        class_target = np.array([-100.] * self.anchors.shape[0], np.int32)

        # pos
        pos_index = np.where(pos == 1)[0]
        pos_num = len(pos_index)
        if pos_num > 0:
            class_target[pos_index] = 1

        # neg
        neg_index = np.where(neg == 1)[0]
        neg_num = len(neg_index)
        class_target[neg_index] = 0
        class_logits = class_target.reshape(-1, 1)
        pos_neg_diff = np.hstack((class_logits, diff))
        return pos_neg_diff

    def plot_anchor_via_centerwh(anchors,bg_w=271,bg_h=271,target_sz_at_271=None):
        bg_c_x=bg_w/2
        bg_c_y=bg_h/2
        abs_anchor=np.stack([[bg_c_x+x-w/2,bg_c_y+y-w/2,bg_c_x+x+w/2,bg_c_y+y+w/2] for x,y,w,h in anchors])
        backgroud=np.zeros((bg_w,bg_h,3))
        for x1,y1,x2,y2 in abs_anchor.astype('int'):
            _=cv2.rectangle(backgroud,(x1,y1),(x2,y2), (0, 255, 255), 1)
        if target_sz_at_271 is not None:
            w,h= target_sz_at_271
            x1,y1=bg_c_x-w/2,bg_c_y-h/2
            x2,y2=bg_c_x+w/2,bg_c_y+h/2
            x1,y1,x2,y2=np.array([x1,y1,x2,y2],dtype='int')
            _=cv2.rectangle(backgroud,(x1,y1),(x2,y2), (255, 0, 255), 1)
        plt.imshow(backgroud)
