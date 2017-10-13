import tensorflow as tf

'''
conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
)
Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
first and fourth element's mean https://stackoverflow.com/questions/34642595/tensorflow-strides-argument
'''

def alexnet(image):
    '''
    Building the alexnet

    Args:
    Image Tensor: shape is [batch_size,width,height,channels]

    Returns:
    poo(l5: the last Tensor in the convolutional component of AlexNet.
    '''
    # first conv
    # 구체적인 스펙은 논문참조
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,96],dtype=tf.float32,
        stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(image,kernel,[1,4,4,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[96],dtype=tf.float32),
        trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)

    #lrn1
    with tf.name_scope('lrn1') as scope:
            lrn1=tf.nn.local_response_normalization(conv1,alpha=1e-4,beta=0.75,depth_radius=5,bias=2.0)

    # pool1
    # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')

    #conv2
    # 논문에서는 2개의 gpu를 사용 conv2에서부터 2개로 나뉘게 되어 48개 이나 한개의 conv2를 가질예정이므로 96으로 수정
    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,96,256],dtype=tf.float32,stddev=1e-1),name='weights')
        # stride에 대해서 따로 언급이 없으나 역산시 1
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)

    #lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2=tf.nn.local_response_normalization(conv2,alpha=1e-4,beta=0.75,depth_radius=5,bias=2.0)

    #pool2
    pool2=tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,256,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0,0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)

    #conv4
    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,384],dtype=tf.float32,stddev=0.1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)

    #pool5
    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
    return pool5
