import vggNet
import imManipulation
import tensorflow as tf
import numpy as np
import scipy.misc as im
import os
import GuassianBlur
import gc

index = 0

batch_size = 1

tf.reset_default_graph()

weights = np.load("modelWeights5.npz")
bn = weights["update_ops"]

sess = tf.Session()

X = tf.placeholder(tf.float32, [None, 256, 256, 3])

C1= tf.placeholder(tf.float32,[None,256, 256, 3])
C2= tf.placeholder(tf.float32,[None,256, 256, 64])
C3= tf.placeholder(tf.float32,[None,128, 128, 128])
C4= tf.placeholder(tf.float32,[None,64, 64, 256])
C5= tf.placeholder(tf.float32,[None,32, 32, 512])

Y1 = tf.placeholder(tf.float32,[None,256,256,2])

def ini_wt(shape):
    initial = tf.truncated_normal(shape,stddev=0.001)
    return tf.Variable(initial)
Wup3 = ini_wt([3,3,128,128])
c3_ = tf.nn.conv2d_transpose(C3,Wup3,[batch_size,256,256,128],strides=[1,2,2,1],padding='SAME')
Wup4 = ini_wt([3,3,256,256])
c4_ = tf.nn.conv2d_transpose(C4,Wup4,[batch_size,256,256,256],strides=[1,4,4,1],padding='SAME')
Wup5 = ini_wt([3,3,512,512])
c5_ = tf.nn.conv2d_transpose(C5,Wup5,[batch_size,256,256,512],strides=[1,8,8,1],padding='SAME')
hypercolumn = tf.concat(3,[C1,C2,c3_,c4_,c5_])
Ws1 = ini_wt([3,3,963,128])
y = tf.nn.conv2d(hypercolumn,Ws1,strides=[1,1,1,1],padding='SAME')
y = tf.nn.batch_normalization(y, offset=True, scale=True, variance_epsilon = 0.001,mean=bn[0],variance=bn[1])
y = tf.nn.relu(y)
Ws2 = ini_wt([3,3,128,64])
y = tf.nn.conv2d(y,Ws2,strides=[1,1,1,1],padding='SAME')
y = tf.nn.batch_normalization(y, offset=True, scale=True, variance_epsilon = 0.001, mean=bn[2],variance=bn[3])
y = tf.nn.relu(y)
Ws3 = ini_wt([3,3,64,2])
y = tf.nn.conv2d(y,Ws3,strides=[1,1,1,1],padding='SAME')
print(y.get_shape())
def guassianBlurLoss(y,Y1):
    y_u = tf.expand_dims(y[:,:,:,0],3)
    Y1_U = tf.expand_dims(Y1[:,:,:,0],3)
    y_v = tf.expand_dims(y[:,:,:,1],3)
    Y1_V = tf.expand_dims(Y1[:,:,:,1],3)

    u1_blur3 = GuassianBlur.guassianBlur(y_u,2)
    u2_blur3 = GuassianBlur.guassianBlur(Y1_U,2)
    v1_blur3 = GuassianBlur.guassianBlur(y_v,2)
    v2_blur3 = GuassianBlur.guassianBlur(Y1_V,2)

    uv1 = tf.concat(3, (u1_blur3,v1_blur3))
    UV1 = tf.concat(3, (u2_blur3, v2_blur3))

    u1_blur5 = GuassianBlur.guassianBlur(y_u,3)
    u2_blur5 = GuassianBlur.guassianBlur(Y1_U,3)
    v1_blur5 = GuassianBlur.guassianBlur(y_v,3)
    v2_blur5 = GuassianBlur.guassianBlur(Y1_V,3)

    uv2 = tf.concat(3,(u1_blur5,v1_blur5))
    UV2 = tf.concat(3, (u2_blur5, v2_blur5))

    l = (tf.reduce_sum(tf.squared_difference(y,Y1))+tf.reduce_sum(tf.squared_difference(uv1,UV1))+tf.reduce_sum(tf.squared_difference(uv2,UV2)))/3
    return l

loss = guassianBlurLoss(y,Y1)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer().minimize(loss)

parameters = [Ws1,Ws2,Ws3,Wup3,Wup4,Wup5]
keys = sorted(weights.keys())
keys = keys[:-1]
for i, k in enumerate(keys):
    sess.run(parameters[i].assign(weights[k]))

vgg = vggNet.vgg16Net(X, '/vggWeights/vgg16_weights.npz', sess)

I = im.imread("temp2.jpg",mode='RGB')
I = I/255
I = np.reshape(I,(1,256,256,3))
c1, c2, c3, c4, c5 = vgg.return_layers(I, sess)
I = np.reshape(I,(256,256,3))

uv_val = sess.run([y], feed_dict={C1: c1, C2: c2, C3: c3, C4: c4, C5: c5})
print(uv_val)
uv_val = np.reshape(uv_val,(256,256,2))
print(uv_val.shape)
u_val = uv_val[:,:,0]
v_val = uv_val[:,:,1]
im.imsave("/output/originalImage.jpg",I)
I = np.reshape(I[:,:,0],(256,256))
r,g,b = imManipulation.yuv2rgb(I,u_val,v_val)
r = np.reshape(r,(256,256,1))
g = np.reshape(g,(256,256,1))
b = np.reshape(b,(256,256,1))
imp = np.concatenate((r,g,b),axis=2)
imp = imp*255
print(imp,end=" ")
im.imsave("/output/colouredImage.jpg",imp)
