import tensorflow as tf
import numpy as np
import OnlineImageReader
import CifarDataSource

class_count=10
learning_rate=0.0001
training_iters=600
dropout = 0.5
useGPU=False
save_dir='./Save'
model_name='model.mod'
prog_name='prog.cfg'

keey_prob = tf.placeholder(tf.float32,name='KeepProb')
input_X = tf.placeholder(tf.float32,shape=[None,32,32,3],name='InputImage')
input_Y = tf.placeholder(tf.float32,shape=[None,class_count],name='ImageLabel')

def constructModel():
    # convert the image data to real data
    with tf.name_scope('InputProc'):
        net=tf.image.convert_image_dtype(input_X, tf.float32,name='Image_to_float')
        net=(net-0.5)*2.
    # add first convolution layer
    with tf.name_scope('CNN_layer_1'):
        net=tf.contrib.layers.convolution2d(net,
                                            num_outputs=64,
                                            kernel_size=(5,5),
                                            weights_initializer=tf.random_normal_initializer(stddev=0.005),
                                            activation_fn=tf.nn.relu,
                                            stride=(1,1),
                                            trainable=True)
        net = tf.nn.max_pool(net,
                           ksize=[1,3,3,1],
                           strides=[1,2,2,1],
                           padding='SAME')
        net = tf.nn.lrn(net,4,bias=1.0,alpha=0.001 / 9.0, beta=0.75)

    # add second convolution layer
    with tf.name_scope('CNN_layer_2'):
        net = tf.contrib.layers.convolution2d(net,
                                            num_outputs=128,
                                            kernel_size=(5,5),
                                            weights_initializer=tf.random_normal_initializer(stddev=0.005),
                                            activation_fn=tf.nn.relu,
                                            stride=(1,1),
                                            trainable=True)
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        net = tf.nn.max_pool(net,
                           ksize=[1,3,3,1],
                           strides=[1,2,2,1],
                           padding='SAME')
    # add third convolution layer
    with tf.name_scope('CNN_layer_3'):
        net = tf.contrib.layers.convolution2d(net,
                                              num_outputs=256,
                                              kernel_size=(5, 5),
                                              weights_initializer=tf.random_normal_initializer(stddev=0.005),
                                              activation_fn=tf.nn.relu,
                                              stride=(4, 4),
                                              trainable=True)
        net = tf.nn.avg_pool(net,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        # flatten the image ready for full connection layer
        net=tf.reshape(net,[-1,256],name='Reshaper')

    # create the first full connection layer
    with tf.name_scope('Relu_full_1'):
        net=tf.contrib.layers.fully_connected(net,
                                              256,
                                              weights_initializer=tf.random_normal_initializer(stddev=0.005),
                                              activation_fn=tf.nn.relu)
        net=tf.nn.dropout(net,keey_prob)
    # create the second full connection layer
    with tf.name_scope('Relu_full_2'):
        net = tf.contrib.layers.fully_connected(net,
                                                128,
                                                weights_initializer=tf.random_normal_initializer(stddev=0.005),
                                                activation_fn=tf.nn.relu)
        net = tf.nn.dropout(net, keey_prob)

    # create the final full connection layer
    with tf.name_scope('Output'):
        net=tf.contrib.layers.fully_connected(net,
                                              class_count,
                                              weights_initializer=tf.random_normal_initializer(stddev=0.005))
    return net

def getBatch(readers,batchSize):
    vec=[]
    lab=[]
    for i in range(0,len(readers)):
        print(len(readers))
        vec.append(readers[i].getBatch(batchSize))

def modelLoadOrInit(sess,saver):
    # return current batch and eval step
    import os
    if not os.path.exists(save_dir):
        print('dir no exists, create save model dir')
        os.mkdir(save_dir)
    if (not os.path.exists(save_dir+'/'+model_name)) or (not os.path.exists(save_dir+'/'+prog_name)):
        # Initializing the variables
        print('model no exists, init all variables')
        init = tf.global_variables_initializer()
        sess.run(init)
        return 0,1
    else:
        print('model exists, load variables from model')
        saver.restore(sess,save_dir+'/'+model_name)
        f=open(save_dir+'/'+prog_name,'r')
        line=f.readline()
        line=line.split(':')
        f.close()
        return int(line[0]),int(line[1])

def saveModel(sess,saver,currentBatch,evalStep):
    print('saving model')
    saver.save(sess,save_dir+'/'+model_name)
    import os
    if os.path.exists(save_dir+'/'+prog_name):
        os.remove(save_dir+'/'+prog_name)
    f=open(save_dir+'/'+model_name,'w')
    f.write(str(currentBatch)+':'+str(evalStep))
    f.close()
    print('model saved')

#read config
cfg=open('trainConfig.cfg')
cfg=cfg.read()
cfg=cfg.split()
for t in range(0,len(cfg)):
    if t=='':
        cfg.remove('')
# create readers from config
'''
readers=[]
for i in range(0,len(cfg)):
    reader=OnlineImageReader.ImageReader(cfg[i].split(',')[0],cfg[i].split(',')[1])
    readers.append(reader)
    print(reader.imageClass()+' : '+str(reader.imageCount()))

getBatch(readers,2)
'''

# cal to construct the model
predict=constructModel()

# Summary writer (Create and write the current graph)
writer=tf.train.SummaryWriter('./Log',graph=tf.get_default_graph())

# define loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,input_Y),name='cost')
#recode the cost
tf.scalar_summary('EvalLoss',cost)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,name='opt')

# Evaluate model
correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(input_Y, 1),name='correcct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='acc')
#recode the acc
tf.scalar_summary('EvalAcc',accuracy)

# Graph drawer
merged = tf.merge_all_summaries()

# Create saver
saver = tf.train.Saver()

sess=tf.Session()
# Load or init model
currentBatch,evalStep = modelLoadOrInit(sess,saver)

# read cifar_10 data
dataSource=CifarDataSource.DataSource()

#evalStep=1
#currentBatch=0
while currentBatch<25:
    print('begin batch '+str(currentBatch)+' training:')
    step =1
    trainImg,trainLab=dataSource.getTrainBatch(2000)
    while step <= training_iters:
        if useGPU:
            with tf.device('/gpu:0'):
                sess.run(optimizer, feed_dict={input_X:trainImg,
                                               input_Y:trainLab,
                                               keey_prob:dropout})
        else:
            sess.run(optimizer, feed_dict={input_X: trainImg,
                                           input_Y: trainLab,
                                           keey_prob: dropout})
        if step % 10 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={input_X:trainImg,
                                                              input_Y:trainLab,
                                                              keey_prob:1.})
            testImg,testLab=dataSource.getTestBatch()
            mergedResult,testAcc=sess.run([merged,accuracy],feed_dict={input_X:testImg,
                                                                       input_Y:testLab,
                                                                       keey_prob:1.})

            writer.add_summary(mergedResult,evalStep)
            writer.flush()
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + ", Eval Accuracy= " + \
                  "{:.5f}".format(testAcc))

        step += 1
        evalStep += 1
    print('one batch done!')
    currentBatch+=1
    saveModel(sess, saver,currentBatch,evalStep)

print("Optimization Finished!")
# Calculate accuracy
testImg,testLab=dataSource.getTestData()
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={input_X:testImg,
                                  input_Y:testLab,
                                  keey_prob:1.}))
writer.close()