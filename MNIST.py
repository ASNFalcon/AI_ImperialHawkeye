# -*- coding: utf-8 -*- #
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
'''
    author: James-J
    time: 2018/10/26
    模型说明：
        一个较为完整的mnist数据集训练模型 全连接模型 精确度接近98%
        使用指数衰减学习率、滑动平均、正则化、模型保存与读取
        过程中需要联网下载mnist数据集
        tensorflow的GPU版本运行 并且设置GPU按需分配
    修改说明：
        使用GPU:0运行程序 如果需要改到CPU请屏蔽刚开始的几行配置信息
        如果需要更改模型保存的位置 自行修改MODEL_SAVE_PATH 这里默认为当前文件夹存储模型文件
'''
 
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU 0
config = tf.ConfigProto()  # 获取配置信息
config.log_device_placement = False  # 不输出设备和tensor详细信息
config.gpu_options.allow_growth=True  # GPU按需分配大小 不然的话会显示GPU占用率很高
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # 减少不必要的输出信息
MODEL_SAVE_PATH = './model/'  # 存储模型的路径 我选择了当前文件所在的路径
MODEL_NAME = 'mnist-model'  # 模型存储的名称
 
#=========================超参数========================
INPUT_SIZE = 784  # 输入数据是mnist的每张图片 28*28 = 784
HIDDEN_SIZE = 512  # 隐藏层自定义 太小的话神经网络训练的效果不好
OUTPUT_SIZE = 10  # 输出层只有0~9这10个类别
BATCH_SIZE = 100  # 一次输入一百张图片来训练
LR_BASE = 0.09  # 基础学习率  使用指数衰减
LR_DECAY = 0.99  # 指数衰减参数
TRAIN_STEPS = 30000  # 总共训练的轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均参数
REGULARIZATION_RATE = 0.0001  # 正则化系数
#=======================================================
 
 
def weight_variable(name, shape, regularizar):
    """
    定义与获取权重变量
    :param name: 变量名
    :param shape: 变量大小
    :param regularizar: 是否使用正则化
    :return: 权重
    """
    weights = tf.get_variable(name, shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
 
    if regularizar != None:
        # 使用正则化时 将所有权重乘以一个正则化系数加入损失函数中 来达到限制权重从而防止过拟合的目的
        tf.add_to_collection('losses',regularizar(weights))
    return weights
 
 
def inference(input_tensor, regularizer):
    """
    前向传播过程 输入层784个节点 隐藏层512个节点 输出层10个节点
    :param input_tensor: 输入的变量
    :param regularizer: 是否使用正则化
    :return: 前向传播结果
    """
    with tf.variable_scope('layer_1'):  # 定义第一层
        weights_1 = weight_variable('weights_1',
                                    [INPUT_SIZE, HIDDEN_SIZE],
                                    regularizer)
        bias_1 = tf.get_variable('bias_1',
                                  [HIDDEN_SIZE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights_1) + bias_1)  # 输入变量乘以权重加上偏置值 最后使用relu函数
 
    with tf.variable_scope('layer_2'):  # 定义第二层
        weights_2 = weight_variable('weights_2',
                                    [HIDDEN_SIZE, OUTPUT_SIZE],
                                    regularizer)
        bias_2 = tf.get_variable('bias_2',
                                  [OUTPUT_SIZE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights_2) + bias_2)
 
    return layer_2
 
 
def train(mnist):
    """
    传入mnist数据集 并用mnist数据集的训练数据训练练模型
    每经过1000轮就使用mnist数据集的验证集进行验证 如果loss比以前的小就存储模型
    :param mnist: 数据集
    """
    with tf.variable_scope('my_mnist', reuse=tf.AUTO_REUSE):  # 总体变量域
        # None表示一次性传入数据组数的大小未知 训练的时候传入的是 BATCH_SIZE
        # 但是其他情况例如验证和测试的情况下可能就不是了 未知的情况下可以默认为None
        X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='x_input')
        Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y_input')
 
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  # 正则化类
        y = inference(X, regularizer)  # 使用正则化
 
        global_step = tf.Variable(0, trainable=False)  # 全局训练的轮数 该变量不可训练
 
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  # 滑动平均类
        variable_averages_op = variable_averages.apply(tf.trainable_variables())  # 应用到所有可以训练的参数上面
 
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(Y, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 正则化后的损失函数
 
        learning_rate = tf.train.exponential_decay(LR_BASE,
                                                   global_step,
                                                   mnist.train.num_examples / BATCH_SIZE,
                                                   LR_DECAY)  # 指数衰减学习率
 
        train_step = tf.train.GradientDescentOptimizer(learning_rate). \
            minimize(cross_entropy_mean, global_step=global_step)  # 优化器
 
        with tf.control_dependencies([train_step, variable_averages_op]):  # 更新滑动平均值到所有可训练的变量中
            train_op = tf.no_op(name='train')
 
        prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))  # 计算准确性
 
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # 定义模型然后存储 保存的模型数量最多一个
 
        with tf.Session(config=config) as sess:  # 启动会话 传入配置信息
            # 如果以前存在训练过的模型文件 就读取文件 若不存在 就初始化各个变量
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('================ reload model ================')
            else:
                sess.run(tf.global_variables_initializer())  # 初始化各个变量
                print('============= create a new model =============')
 
            # 如果模型内有loss文件 那么读取出已经存进去的loss值 没有就先写一个比较大的数(例如10)进去
            if os.path.isfile(os.path.join(MODEL_SAVE_PATH, './loss.txt')):
                f = open(os.path.join(MODEL_SAVE_PATH, './loss.txt'), 'r')
                loss_save = float(f.read())
                f.close
            else:
                loss_save = 10
                f = open(os.path.join(MODEL_SAVE_PATH, './loss.txt'), 'w')
                f.write(str(loss_save))
                f.close
 
            # 开始训练
            for i in range(TRAIN_STEPS):
                x_input, y_input = mnist.train.next_batch(BATCH_SIZE)  # 获取训练集
                _, loss_value_train, step = sess.run([train_op, loss, global_step],
                                                     feed_dict={X: x_input, Y: y_input})
                if i % 1000 == 0:  # 每1000轮过后就用验证集验证一下
                    val_x = mnist.validation.images
                    val_y = mnist.validation.labels
                    accuracy_, loss_value = sess.run([accuracy, loss],
                                                     feed_dict={X: val_x, Y: val_y})
                    print('After %d training steps, loss = %f , accuracy = %f' % (i, loss_value, accuracy_))
                    # 每1000轮判断是否存储模型
                    if loss_value < loss_save:
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                        print('save model')
                        loss_save = loss_value
                        f = open(os.path.join(MODEL_SAVE_PATH, './loss.txt'), 'w')
                        f.write(str(loss_save))
                        f.close
 
 
def prediction(mnist):
    """
    加载已经训练好的模型 得到预测结果并求出最终的准确率
    :param mnist: 传入数据 使用测试数据集
    """
    tf.reset_default_graph()  # 由于train和prediction都使用了变量环境'my_mnist' 要将先前残留的变量清除掉
    with tf.variable_scope('my_mnist', reuse=tf.AUTO_REUSE):
        X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='x_input')
        Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y_input')
        y = inference(X, None)  # 这里训练时不适用正则化 因为不需要更新权重了
        prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))  # 计算准确性
 
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(Y, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算loss
 
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_averages_restore = variable_averages.variables_to_restore()  # 选择滑动平均变量读取出来
 
        saver = tf.train.Saver(variable_averages_restore)  # 读取出来的变量是滑动平均的结果
 
        # 启动会话 使用测试数据集传入得到结果
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                loss, accuracy_ = sess.run([cross_entropy_mean, accuracy],
                                           feed_dict={X: mnist.test.images, Y: mnist.test.labels})
                print('Test accuracy = %f' % (accuracy_))
            else:
                print('No found model')
 
 
if __name__ == '__main__':
    # mnist数据集将会下载在当前文件夹新建的MNIST_data文件夹中
    # one_hot=True标签值只有一个是1 其余都是0 原先的值是0~1之间的数
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)
    prediction(mnist)
    
