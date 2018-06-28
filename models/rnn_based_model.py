"""
@author - Sivaramakrishnan
"""
import tensorflow as tf

class RNN_model(object):
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

    def RNN(self,x,w,b):
        lstm = tf.contrib.rnn.BasicLSTMCell(64)
        #state = tf.zeros([self.model_dict["batch_size"],64])
        state = lstm.zero_state(self.model_dict["batch_size"],dtype=tf.float32)
        label = tf.zeros([self.model_dict["batch_size"],1])
        labels = []
        for i in range(self.model_dict["num_of_steps"]):
            input_ = tf.concat([label,x],axis = 1)
            label,state = lstm(input_,state)
            label = tf.matmul(label,w)+b
            labels.append(label)
        return tf.reshape(tf.squeeze(labels,axis = -1),[self.model_dict["batch_size"],-1])

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

            with tf.name_scope("rnn_1"):
                rnn_1_w = self.weights_("rnn_1w",[64,1])
                rnn_1_b = self.biases_("rnn_1b",[1])

        with tf.device(self.model_dict["devices"][1]):
            conv_1 = self.conv_max_pool(self.X,conv_1_w,conv_1_b)
            conv_2 = self.conv_max_pool(conv_1,conv_2_w,conv_2_b)
            reshaped = tf.reshape(conv_2,[-1,56*56*64])
            full_1 = tf.nn.relu(tf.add(tf.matmul(reshaped,full_1_w),full_1_b))
            #out = tf.add(tf.matmul(full_1,full_2_w),full_2_b)

            out = self.RNN(full_1,rnn_1_w,rnn_1_b)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = out,labels = self.Y))

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(self.model_dict["lr"]).minimize(self.loss)

        with tf.name_scope("accuracy"):
            self.predictions = tf.cast(tf.greater_equal(out,0.5),tf.float32)
            self.accuracy = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.equal(self.predictions,self.Y),tf.float32),axis = 1))

        return [self.loss,self.optimizer,self.predictions]
