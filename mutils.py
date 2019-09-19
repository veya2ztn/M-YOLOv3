import torch
from torch.autograd import Variable

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def generate_new_kernels(pool,classes,level,in_channel,cuda):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    kernel_shape=(15,in_channel,1,1)
    bias_shape=15
    level=str(level)
    kernels=[]
    biases =[]
    kname  ='k'+level
    bname  ='b'+level
    for _class in classes:
        kernel=torch.zeros(kernel_shape)
        _=torch.nn.init.normal_(kernel)
        kernel=Variable(kernel.type(Tensor),name=_class+'.'+kname)

        bias=torch.zeros(bias_shape)
        _=torch.nn.init.normal_(bias)
        bias=Variable(bias.type(Tensor),name=_class+'.'+bname)

        if _class not in pool:pool[_class]={}
        pool[_class][kname]=kernel
        pool[_class][bname]=bias
        kernels.append(kernel)
        biases.append(bias)
    return (kernels,biases)


def review_plug_in(pool,classes,cuda):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    assert set(classes)>=set(pool.keys())
    def SV(tensor,name,Tensor):return Variable(tensor.type(Tensor),name=name)
    k1=[SV(pool[_class]['k1'],_class+'.'+'k1',Tensor) for _class in classes]
    b1=[SV(pool[_class]['b1'],_class+'.'+'b1',Tensor) for _class in classes]
    k2=[SV(pool[_class]['k2'],_class+'.'+'k2',Tensor) for _class in classes]
    b2=[SV(pool[_class]['b2'],_class+'.'+'b2',Tensor) for _class in classes]
    k3=[SV(pool[_class]['k3'],_class+'.'+'k3',Tensor) for _class in classes]
    b3=[SV(pool[_class]['b3'],_class+'.'+'b3',Tensor) for _class in classes]
    return [(k1,b1),
            (k2,b2),
            (k3,b3)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.win100=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        #self.avg = self.sum / self.count
        self.win100.append(val)
        if len(self.win100) > 100:_=self.win100.pop(0)
        sum100=sum(self.win100)
        self.avg=1.0*sum100/len(self.win100)
import copy
class RecordLoss:

    def __init__(self,init_loss_records,init_loss_show,x_window):
        self.loss_records=init_loss_records
        self.init_loss_show=copy.deepcopy(init_loss_show)
        self.loss_showeds=copy.deepcopy(init_loss_show)
        self.x_window = x_window
        self.x=[]
        self.x_bounds=[0,x_window]
        self.y_bounds=[0,100]
    def record_loss(self,loss_recorder):
        return loss_recorder.avg

    def update_record(self,recorder,loss):
        recorder.update(loss.cpu().item())

    def update(self,loss_list):
        for loss_recorder,loss_showed,loss in zip(self.loss_records,self.loss_showeds,loss_list):
            self.update_record(loss_recorder,loss)
            if loss_showed is not None:
                loss_showed.append(self.record_loss(loss_recorder))
    def reset(self):
        self.x=[]
        self.loss_showeds=copy.deepcopy(self.init_loss_show)

    def step(self,step,loss_list):
        x_window=self.x_window
        if step >x_window and step%x_window==1:
            graphs=[show for show in self.loss_showeds if show is not None]
            self.x_bounds = [0, self.x_window]
            self.y_window = max(max(graphs))-min(min(graphs))
            now_at        = self.record_loss(self.loss_records[0])
            self.y_bounds = [now_at-self.y_window, now_at+0.2*self.y_window]
            self.reset()
        self.x.append(step%x_window)
        self.update(loss_list)


    def update_graph(self,mb,step):
        if step >self.x_window:
            graphs=[show for show in self.loss_showeds if show is not None]
            graphs=[[self.x]+graphs]
            x_bounds=self.x_bounds
            y_bounds=self.y_bounds
            mb.update_graph(graphs, x_bounds, y_bounds)
            #return ll

    def print2file(self,step,file_name):
        with open(file_name,'a') as log_file:
            ll=["{:.4f}".format(self.record_loss(recorder)) for recorder in self.loss_records]
            printedstring=str(step)+' '+' '.join(ll)+"\n"
            #print(printedstring)
            _=log_file.write(printedstring)
