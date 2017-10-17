# alexnet
___

## about

The neural network, which has 60 million parameters and 650,000 neurons, consists
of five convolutional layers, some of which are followed by max-pooling layers,
and three fully-connected layers with a final 1000-way softmax. To make training
faster, we used non-saturating neurons and a very efficient GPU implementation
of the convolution operation. To reduce overfitting in the fully-connected
layers we employed a recently-developed regularization method called “dropout”
that proved to be very effective.

## architecture
![](https://kratzert.github.io/images/finetune_alexnet/alexnet.png)

## requirement

* tensorflow-gpu (ver.1.3.1)
* cv2 (ver.3.3.0)
* numpy (ver 1.13.3)
* scipy (ver 0.19.1)


## Usage
1. Download the image file from the link below.(LSVRC2012 train,val,test,Development kit (Task 1))
1. untar.(There is a script in `etc`)
1.Modify  `IMAGENET_PATH` in train.py hyperparameter(maybe you need).

## train

### From the beginning

```
python3 train.py
```

### resume training

```
python3 train.py -resume
```

## test

```
python3 test.py
```

## Classify


## tensorboard

```
tensorboard --logdir path/to/summary/train/
```

![](https://galoismilk.org/storage/etc/graph-large_attrs_key=_too_large_attrs&limit_attr_size=1024&run=.png)


## TODO

* ~~apply another optimizer ~~
* ~~apply tensorboard ~~
* ~~Fit to a GPU~~
* ~~Application of the technique to the paper~~
* Eliminate bottlenecks



## batch normaliztion

[batch normaliztion](https://arxiv.org/abs/1502.03167)is decreasing technical skill,Gradient Vanishing & Gradient Exploding
![](http://nmhkahn.github.io/assets/Casestudy-CNN/alex-norm1.png)


### k=2,n=5,α=10−4,β=0.75k=2,n=5,α=10−4,β=0.75

 ![](https://shuuki4.files.wordpress.com/2016/01/bn1.png)
 ![](https://shuuki4.files.wordpress.com/2016/01/bn2.png)

## file_architecture

```
ILSVRC 2012 training set folder should be srtuctured like this:
		ILSVRC2012_img_train
			|_n01440764
			|_n01443537
			|_n01484850
			|_n01491361
			|_ ...
```    

### you must untar training file `untar.sh`


## download

[download LSVRC 2012 image data file](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads)

## optimizer

Apply AdamOptimizer
![](http://i.imgur.com/2dKCQHh.gif?1)
![](http://i.imgur.com/pD0hWu5.gif?1)
![](http://i.imgur.com/NKsFHJb.gif?1)

## Remove log

If you do not want to see the log at startup
train.py line 97, remove `allow_soft_placement=True, log_device_placement=True`

## references

[optimizer](http://ruder.io/optimizing-gradient-descent/)

[AlexNet training on ImageNet LSVRC 2012](https://github.com/dontfollowmeimcrazy/imagenet)

[Tensorflow Models](https://github.com/tensorflow/models)

[Tensorflow API](https://www.tensorflow.org/versions/r1.2/api_docs/)

## Licence

[MIT Licence](LICENSE)
