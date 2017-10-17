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

## Usage
1. 아래 다운로드 링크에서 자료를 다운 받는다.(LSVRC2012 train,val,test,Development kit (Task 1))
1. 다운받은 파일을 untar한다.(etc에 스크립트 존재)
1. train.py에서 `IMAGENET_PATH`와 필요하다면 hyperparameter를 수정한다.

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

## tensorboard

```
tensorboard --logdir path/to/summary/train/
```
![](https://galoismilk.org/storage/etc/graph-large_attrs_key=_too_large_attrs&limit_attr_size=1024&run=.png)


## TODO
* ~~another optimizer 적용~~
* ~~tensorboard 적용~~
* ~~GPU 1개를 이용하여 최적화~~
* ~~논문에 나온 최적화기법 모두 적용~~


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

### train파일은 그냥 untar하면 안되며 untar.sh를이용하여 폴더를 만든후 untar해야한다.


## download
[download LSVRC 2012 image data file](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads)

## optimizer
이중에서 adam 적용
![](http://i.imgur.com/2dKCQHh.gif?1)
![](http://i.imgur.com/pD0hWu5.gif?1)
![](http://i.imgur.com/NKsFHJb.gif?1)



## references

[optimizer](http://ruder.io/optimizing-gradient-descent/)
[AlexNet training on ImageNet LSVRC 2012](https://github.com/dontfollowmeimcrazy/imagenet)
[Tensorflow Models](https://github.com/tensorflow/models)
[Tensorflow API](https://www.tensorflow.org/versions/r1.2/api_docs/)
