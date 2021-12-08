# coding: utf-8

import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TCNNConfig(object):
    embedding_dim = 64  # 詞向量維度
    seq_length = 600  # 序列長度
    num_classes = 6  # 類別數
    num_filters = 256  # 卷積核數目
    # kernel_sizes = [3, 4, 5]  # 卷積核尺寸
    kernel_size = 5 # 卷積核尺寸
    vocab_size = 140  # 詞彙表大小
    hidden_dim = 256  # 全連接層神經元
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 學習率
    batch_size = 64  # 每批訓練大小
    num_epochs = 10  # 總迭代輪次
    print_per_batch = 20  # 每多少輪輸出一次結果
    save_per_batch = 10  # 每多少輪存入tensorboard

class TextCNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):
            embedding = tf.compat.v1.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')  # global max pooling layer
        '''
        for kernel_size in self.config.kernel_sizes:
            gmps = []
            with tf.name_scope("cnn-%s" % kernel_size):
                # CNN layer
                conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, kernel_size)
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1])
                gmps.append(gmp)
        gmp = tf.concat(values=gmps, name='last_pool_layer', axis=3)
        '''
        with tf.name_scope("score"):
            # 全連接層，後面跟dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # 分類器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 預測類別
        with tf.name_scope("optimize"):
            # 損失函數，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)  # 優化器
        with tf.name_scope("accuracy"):
            # 準確率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''
cnn模型中，首先定義一個一維卷積層，再使用tf.reduce_max實現global max pooling。
再接兩個dense層分別做映射和分類。使用交叉熵損失函數，Adam優化器，並且計算準確率。
這裡有許多參數可調，大部分可以通過調整TCNNConfig類即可。

處理文字任務流程都是：
1、處理數據集，建立詞表，建立詞彙和數字的映射表
2、詞嵌入 把句子轉化為數字
3、思考用什麼模型，把經過轉化為數字的句子 加入到模型
4、建立優化器
5、訓練
'''
