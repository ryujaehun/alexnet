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

## TODO
* another optimizer 적용
* tensorboard 적용
* GPU 1개를 이용하여 최적화
* 논문에 나온 최적화기법 모두 적용


## batch normaliztion

[batch normaliztion](https://arxiv.org/abs/1502.03167)is decreasing technical skill,Gradient Vanishing & Gradient Exploding
![](http://nmhkahn.github.io/assets/Casestudy-CNN/alex-norm1.png)


 $${k=2,n=5,α=10−4,β=0.75k=2,n=5,α=10−4,β=0.75}$$

 ![](https://shuuki4.files.wordpress.com/2016/01/bn1.png)
 ![](https://shuuki4.files.wordpress.com/2016/01/bn2.png)

## download
download [image](http://www.image-net.org/challenges/LSVRC/2010/download-all-nonpub)&[list of images](http://www.image-net.org/download-imageurls)

## optimizer
![](http://i.imgur.com/2dKCQHh.gif?1)
![](http://i.imgur.com/pD0hWu5.gif?1)
![](http://i.imgur.com/NKsFHJb.gif?1)


## references

[optimizer](http://ruder.io/optimizing-gradient-descent/)
[AlexNet training on ImageNet LSVRC 2012](https://github.com/dontfollowmeimcrazy/imagenet)
[Tensorflow Models](https://github.com/tensorflow/models)
[Tensorflow API](https://www.tensorflow.org/versions/r1.2/api_docs/)
