# Plug-in YOLOv3

The layer before every YOLOlayer in YOLOv3 is a naive convolution layer. 

The output for this convolution layer is tensor (B,nA*(nP+nCof+nCls),W,H).

Here

```PYTHON
#nA is the number of anchor
#nP is the number of shift position - {x,y,w,h} per fearture map (W x H) pixel
#nCof is the confidence channel, set nCof = 1
#nCls is the class channel, default set nCls = 80
```

Due to the linear property of convolution operation:
$$
A*(B+C)=A*B+A*C \\
A*([B,C])=[A*B,A*C]
$$
we can simple divide the weight of convolution layer into its corresponding knowledge. That is 

  A input tensor (B,C,W,H) from Resnet Feature Extract Network convolute with 

  the weight tenrsor (C, nA*(nP+nCof+nCls),1,1) then get

  the result tensor (B,nA*(nP+nCof+nCls),W,H).

Can be regard as input tensor convolute with a series of weights 

(C, nA\*[nP, nCof, 1, 1, 1,....., 1], 1, 1)

Based on this, we can generate a continual learning framework.

- The head netword (uausally a resnet network) is a feature extract network, which will convert the input image to a high dimension **feature** ( B, big C, W,H ). 
  - In order to get the reponse strength (other word, divide background and foreground), we convolute **feature** and **response weight** (C, nA\*nCof, 1, 1), get the confidence channel of Yolo.
  - In order to get the four corner shift per pixel, we convolute **feature** and **corner weight** (C, nA\*nP, 1, 1), get the position channel of Yolo.
  - In order to identify whether the detected region belong to **class A**, we convolute **feature** and **class A weight** (C, nA\*1, 1, 1), get the position channel of Yolo.
  - In order to identify whether the detected region belong to **class B**, we convolute **feature** and **class B weight** (C, nA\*1, 1, 1), get the position channel of Yolo.
  - .......

This repository finish this job.

**Acknowledge:** this repository is based on the [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

------------

When we need to train a new knowledge, that is, want to identify whether the detected region belong to **class N**, we only need to train a **class B weight** (C, nA\*1, 1, 1) while we don't change the performance on previous **classes**. 

It can be imaged that the performance for new class classifier is limited, since we can only train a little parameter for fit it. This performance quite depend on the universality of head network. Since we don't train the **response weight** and **corner weight** since we don't want to effect the performance on original taskes.

-----

Basicly, the classification math rule behind yolo is first train the P(is or not) then train P(what class is|it is). So the train for classes and corner only works when area is responsed.

If we can design a better bit-wise trainning stratage for **confidence**, so that every time add new knowledge requirement, the old knowledge learned will not be forgotten. <u>A very beautiful continual learning achieve!</u>





