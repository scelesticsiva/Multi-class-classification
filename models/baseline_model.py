"""
@author - Sivaramakrishnan
"""
import tensorflow as tf

class model(object):
    def __init__(self,X,Y,model_dict):
        self.X = X
        self.Y = Y
        self.model_dict = model_dict

    def weights_(self,name,shape,initializer = tf.contrib.layers.xavier_initializer()):
        return tf.get_variable(name = name,shape = shape,initializer = initializer)

    def biases_(self,name,shape,initializer = tf.constant_initializer(0.01)):
        return tf.get_variable(name = name,shape = shape,initializer = initializer)

    def conv_2d(self,inp,weights,strides = 1):
        return tf.nn.conv2d(inp,weights,strides = [1,strides,strides,1],padding = "SAME")

    def max_pool_2d(self,inp,kernel = 3,strides = 2):
        return tf.nn.max_pool(inp,ksize = [1,kernel,kernel,1],strides = [1,strides,strides,1],padding = "SAME")

    def conv_max_pool(self,inp,w,b):
        conv = tf.nn.relu(tf.nn.bias_add(self.conv_2d(inp,w),b))
        return self.max_pool_2d(conv)

    def inference(self):
        with tf.device(self.model_dict["devices"][0]):
            with tf.name_scope("conv_1"):
                conv_1_w = self.weights_("conv_1w",[3,3,1,32])
                conv_1_b = self.biases_("conv_1b",[32])

            with tf.name_scope("conv_2"):
                conv_2_w = self.weights_("conv_2w",[3,3,32,64])
                conv_2_b = self.biases_("conv_2b",[64])

            with tf.name_scope("full_1"):
                full_1_w = self.weights_("full_1w",[56*56*64,128])
                full_1_b = self.biases_("full_1b",[128])

            with tf.name_scope("full_2"):
                full_2_w = self.weights_("full_2w",[128,15])
                full_2_b = self.biases_("full_2b",[15])

        with tf.device(self.model_dict["devices"][1]):
            conv_1 = self.conv_max_pool(self.X,conv_1_w,conv_1_b)
            conv_2 = self.conv_max_pool(conv_1,conv_2_w,conv_2_b)
            reshaped = tf.reshape(conv_2,[-1,56*56*64])
            full_1 = tf.nn.relu(tf.add(tf.matmul(reshaped,full_1_w),full_1_b))
            out = tf.add(tf.matmul(full_1,full_2_w),full_2_b)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.Y,logits = out))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(self.model_dict["lr"]).minimize(self.loss)


        return self.optimizer