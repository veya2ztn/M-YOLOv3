from __future__ import division

# %load train.py
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from config import myolo_config
from mutils import *
import os
opt=myolo_config("/media/tianning/DATA/COCO/data")
opt.batch_size=10
# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = opt.train

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("yolotrain_output", exist_ok=True)
os.makedirs("yolotrain_checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Initiate model
model =MDarknet('config/m-yolov3.cfg')
#model =Darknet('config/yolov3.cfg')
#model.load_weights("myolo_checkpoints/0.weights")
model.apply(weights_init_normal)
if cuda:model = model.cuda()
#model.train();

model.train()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

cls_k_b_pool={}
feed_cls_k_b=[generate_new_kernels(cls_k_b_pool,classes,1,1024,cuda),
              generate_new_kernels(cls_k_b_pool,classes,2,512,cuda),
              generate_new_kernels(cls_k_b_pool,classes,3,256,cuda)]

all_kernel_param=[x for l in feed_cls_k_b for kb in l for x in kb]
print(cls_k_b_pool.keys())
print(cls_k_b_pool['person'].keys())
# Get hyper parameters
hyperparams   = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum      = float(hyperparams["momentum"])
decay         = float(hyperparams["decay"])
burn_in       = int(hyperparams["burn_in"])

lr=0.0001
# optimizer = torch.optim.Adam([
#                 {'params': filter(lambda p: p.requires_grad, model.parameters()),'lr': learning_rate},
#                  {'params': all_kernel_param, 'lr': learning_rate}
#             ])
i=0

optimizer = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.parameters())},
                 {'params': all_kernel_param}
            ],lr=lr, momentum=momentum, weight_decay = decay)

#from train_utils import *
losses = AverageMeter()
log_hand=RecordLoss([losses],[[]],100)
# mb = master_bar(range(10))
# mb.names = ['tloss']
from tqdm import tqdm
epoch_bar=tqdm(range(5))
batch_bar=tqdm(total=len(dataloader))
for batch_i, (_, imgs, targets) in enumerate(dataloader):break
print("=============start train===============")
steps=0
for epoch in epoch_bar:
    cur_lr = adjust_learning_rate(lr, optimizer, epoch, gamma=0.1)
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        loss = model([imgs,feed_cls_k_b], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if np.isnan(loss.cpu() .detach().numpy()):sys.exit(0)
        log_hand.step(steps,[loss])
        #log_hand.update_graph(mb,steps)
        tqdm.write("Losses:total {:.2f}, conf {:.4f}, r: {:.1f}"
                    .format(
                        loss.item(),
                        model.losses["conf"],
                        model.losses["recall"],
                    )
                    )
        # print(
        #     "Epoch {}/{}, Batch {}/{} Losses:total {:.2f}, conf {:.4f}, r: {:.1f}"
        #     .format(
        #         epoch,
        #         opt.epochs,
        #         batch_i,
        #         len(dataloader),
        #         loss.item(),
        #         model.losses["conf"],
        #         model.losses["recall"],
        #     ), end="\r")

        model.seen += imgs.size(0)
        steps += 1
        batch_bar.update(1)
        if steps%10==1:
            log_hand.print2file(steps,os.path.join(opt.checkpoint_dir,"loss.log"))
    if epoch % opt.checkpoint_interval == 0:
            model.save_weights("%s/epoch_%d.weights" % (opt.checkpoint_dir, epoch))
            torch.save(cls_k_b_pool,"%s/epoch_%d.plug_in_80_class.weight"% (opt.checkpoint_dir, epoch))
