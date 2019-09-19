import numpy as np
import os

class myolo_config:
    epochs             =30

    batch_size         =2
    model_config_path  ="config/yolov3.cfg"
    data_config_path   ="config/coco.data"
    weights_path       ="weights/yolov3.weights"
    conf_thres         =0.8
    nms_thres          =0.4
    n_cpu              =0
    img_size           =416
    checkpoint_interval=1
    checkpoint_dir     ="checkpoints"
    use_cuda           =True
    classes= 80
    backup="backup/"
    def __init__(self,dir):
        self.image_folder =os.path.join(dir,"samples")
        self.class_path   =os.path.join(dir,"coco.names")
        self.train        =os.path.join(dir,"coco/trainvalno5k.txt")
        self.valid        =os.path.join(dir,"coco/5k.txt")
        self.names        =os.path.join(dir,"coco.names")
