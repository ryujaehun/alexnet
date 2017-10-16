import tensorflow as tf
import util as tu


def alexnet(image):
'''
Building the alexnet

Args:
Image Tensor: shape is [batch_size,width,height,channels]

Returns:
pool5: the last Tensor in the convolutional component of AlexNet.
'''
with tf.name_scope('alexnet') as scope:
    with tf.name_scope('conv1') as inner_scope:
        wcnn1 = tu.weight([11, 11, 3, 96], name='cnn1')
        bcnn1 = tu.bias(0.0, [96], name='cnn1')
        conv1 = tf.add(tu.conv2d(x, wcnn1, stride=(4, 4), padding='SAME'), bcnn1)
        #conv1 = tu.batch_norm(conv1)
        conv1 = tu.relu(conv1)
        norm1 = tu.lrn(conv1, depth_radius=5, bias=2.0, alpha=1e-04, beta=0.75)
        pool1 = tu.max_pool2d(norm1, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

    with tf.name_scope('conv2') as inner_scope:
        wcnn2 = tu.weight([5, 5, 96, 256], name='wcnn2')
        bcnn2 = tu.bias(1.0, [256], name='bcnn2')
        conv2 = tf.add(tu.conv2d(pool1, wcnn2, stride=(1, 1), padding='SAME'), bcnn2)
        #conv2 = tu.batch_norm(conv2)
        conv2 = tu.relu(conv2)
        norm2 = tu.lrn(conv2, depth_radius=5, bias=2.0, alpha=1e-04, beta=0.75)
        pool2 = tu.max_pool2d(norm2, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

    with tf.name_scope('conv3') as inner_scope:
        wcnn3 = tu.weight([3, 3, 256, 384], name='wcnn3')
        bcnn3 = tu.bias(0.0, [384], name='bcnn3')
        conv3 = tf.add(tu.conv2d(pool2, wcnn3, stride=(1, 1), padding='SAME'), bcnn3)
        #conv3 = tu.batch_norm(conv3)
        conv3 = tu.relu(conv3)

    with tf.name_scope('alexnet_cnn_conv4') as inner_scope:
        wcnn4 = tu.weight([3, 3, 384, 384], name='wcnn4')
        bcnn4 = tu.bias(0.0, [384], name='bcnn4')
        conv4 = tf.add(tu.conv2d(conv3, wcnn4, stride=(1, 1), padding='SAME'), bcnn4)
        #conv4 = tu.batch_norm(conv4)
        conv4 = tu.relu(conv4)

    with tf.name_scope('alexnet_cnn_conv5') as inner_scope:
        wcnn5 = tu.weight([3, 3, 384, 256], name='wcnn5')
        bcnn5 = tu.bias(0.0, [256], name='bcnn5')
        conv5 = tf.add(tu.conv2d(conv4, wcnn5, stride=(1, 1), padding='SAME'), bcnn5)
        #conv5 = tu.batch_norm(conv5)
        conv5 = tu.relu(conv5)
        pool5 = tu.max_pool2d(conv5, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

    return pool5

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

        # fully-connected6

        flattened=tf.reshape(pool5,[-1,6*6*256])
        fc6=self.fc(input=flattened,num_in=6*6*256,num_out=4096,name='fc6',drop_ratio=1.0-self.keep_prob)

        # fully-connected6
        fc7=self.fc(input=fc6,num_in=6*6*256,num_out=4096,name='fc7',drop_ratio=1.0-self.keep_prob)

        # fully-connected6
        self.fc8=self.fc(input=fc7,num_in=6*6*256,num_out=self.num_classes,name='fc8',drop_ratio=1.0-self.keep_prob)

    def fc(self, input,num_in,num_out,name,drop_ratio=0.5):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',shape = [num_in,num_out],trainable=True)
            biases = tf.get_variable('biases',[num_out],trainable=True)
            # Linear
            act = tf.nn.xw_plus_b(input,weights,biases,name=scope.name)
            relu = tf.nn.relu(act)
            if drop_ratio == 0:
                return relu
            else:
                return tf.nn.dropout(relu,1.0-drop_ratio)
'''

def classifier(x, dropout):
	"""
	AlexNet fully connected layers definition

	Args:
		x: tensor of shape [batch_size, width, height, channels]
		dropout: probability of non dropping out units

	Returns:
		fc3: 1000 linear tensor taken just before applying the softmax operation
			it is needed to feed it to tf.softmax_cross_entropy_with_logits()
		softmax: 1000 linear tensor representing the output probabilities of the image to classify

	"""

	pool5 = alexnet(x)

	dim = pool5.get_shape().as_list()
	flat_dim = dim[1] * dim[2] * dim[3] # 6 * 6 * 256
	flat = tf.reshape(pool5, [-1, flat_dim])

	with tf.name_scope('alexnet_classifier') as scope:
		with tf.name_scope('alexnet_classifier_fc1') as inner_scope:
			wfc1 = tu.weight([flat_dim, 4096], name='wfc1')
			bfc1 = tu.bias(0.0, [4096], name='bfc1')
			fc1 = tf.add(tf.matmul(flat, wfc1), bfc1)
			# fc1 = tu.batch_norm(fc1)
			fc1 = tu.relu(fc1)
			fc1 = tf.nn.dropout(fc1, dropout)

		with tf.name_scope('alexnet_classifier_fc2') as inner_scope:
			wfc2 = tu.weight([4096, 4096], name='wfc2')
			bfc2 = tu.bias(0.0, [4096], name='bfc2')
			fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)
			#fc2 = tu.batch_norm(fc2)
			fc2 = tu.relu(fc2)
			fc2 = tf.nn.dropout(fc2, dropout)

		with tf.name_scope('alexnet_classifier_output') as inner_scope:
			wfc3 = tu.weight([4096, 1000], name='wfc3')
			bfc3 = tu.bias(0.0, [1000], name='bfc3')
			fc3 = tf.add(tf.matmul(fc2, wfc3), bfc3)
			softmax = tf.nn.softmax(fc3)

	return fc3, softmax
