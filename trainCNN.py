import vggNet
import imManipulation
import tensorflow as tf
import numpy as np
import scipy.misc as im
import os

# For Creating channels.npz
# files = os.listdir("dataset")
# i = np.ndarray([1,256,256,1])
# u = np.ndarray([1,256,256,1])
# v = np.ndarray([1,256,256,1])
# for f in files:
#     image = im.imread("dataset/"+f,mode='RGB')
#     I, U, V = imManipulation.rgb2yuv(image)
#     # print (u.shape)
#     I = np.reshape(I, (1,256, 256,1))
#     U = np.reshape(U, (1,256, 256,1))
#     V = np.reshape(V, (1,256, 256,1))
#     i = np.vstack((i,I))
#     u = np.vstack((u,U))
#     v = np.vstack((v,V))
# i = i[1:,:,:]
# u = u[1:,:,:]
# v = v[1:,:,:]
# np.savez("channels.npz",i=i,u=u,v=v) # 1091,256,256,1 each i,u,v

# # For loading channels.npz
# with np.load("channels.npz") as g:
#     I = g['i']
#     U = g['u']
#     V = g['v']

sess = tf.Session()

X = tf.placeholder(tf.float32,[None,256,256,3])
# y = tf.placeholder(tf.float32,[None,256,256,2])

vgg = vggNet.vgg16Net(X, 'vgg16_weights.npz', sess)

image = im.imread("bea1.jpg")
I,U,V = imManipulation.rgb2yuv(image)
I = np.reshape(I, (1,256, 256,1))
U = np.reshape(U, (1,256, 256,1))
V = np.reshape(V, (1,256, 256,1))
print(I.shape)
I = np.concatenate((I, I, I), axis=3)  # 1091,256,256,3
UV = np.concatenate((U,V), axis=3)
print("Intensity layer shape ",I.shape)
c1, c2, c3, c4, c5 = vgg.return_layers(I,sess)
print("Extracted Layers Shape ")
print(c1.shape, c2.shape, c3.shape, c4.shape, c5.shape)

sess.close()

sess = tf.Session()

C1= tf.placeholder(tf.float32,[None,256, 256, 3])
C2= tf.placeholder(tf.float32,[None,256, 256, 64])
C3= tf.placeholder(tf.float32,[None,128, 128, 128])
C4= tf.placeholder(tf.float32,[None,64, 64, 256])
C5= tf.placeholder(tf.float32,[None,32, 32, 512])
Y1 = tf.placeholder(tf.float32,[None,256,256,2])

def ini_wt(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# Make model (not Sequencial) and pass diff conv layers, then merge layers, then shallow net
# model = Model(inputs=[C1,C2,C3,C4,C5],outputs=[y])

Wup3 = ini_wt([3,3,128,128])
c3_ = tf.nn.conv2d_transpose(C3,Wup3,[1,256,256,128],strides=[1,2,2,1],padding='SAME')
Wup41 = ini_wt([3,3,256,256])
c4_ = tf.nn.conv2d_transpose(C4,Wup41,[1,128,128,256],strides=[1,2,2,1],padding='SAME')
Wup42 = ini_wt([3,3,256,256])
c4_ = tf.nn.conv2d_transpose(c4_,Wup42,[1,256,256,256],strides=[1,2,2,1],padding='SAME')
Wup51 = ini_wt([3,3,512,512])
c5_ = tf.nn.conv2d_transpose(C5,Wup51,[1,64,64,512],strides=[1,2,2,1],padding='SAME')
Wup52 = ini_wt([3,3,512,512])
c5_ = tf.nn.conv2d_transpose(c5_,Wup52,[1,128,128,512],strides=[1,2,2,1],padding='SAME')
Wup53 = ini_wt([3,3,512,512])
c5_ = tf.nn.conv2d_transpose(c5_,Wup53,[1,256,256,512],strides=[1,2,2,1],padding='SAME')

hypercolumn = tf.concat([C1,C2,c3_,c4_,c5_],3)
print (" hypercolumn shape : ",hypercolumn.get_shape())

Ws1 = ini_wt([3,3,963,128])
y = tf.nn.conv2d(hypercolumn,Ws1,strides=[1,1,1,1],padding='SAME')
Ws2 = ini_wt([3,3,128,64])
y = tf.nn.conv2d(y,Ws2,strides=[1,1,1,1],padding='SAME')
Ws3 = ini_wt([3,3,64,2])
y = tf.nn.conv2d(y,Ws3,strides=[1,1,1,1],padding='SAME')

# model.compile(optimizer='adam',loss='mean_squared_error')
loss = tf.reduce_mean(tf.squared_difference(y,Y1))
optimizer = tf.train.AdamOptimizer().minimize(loss)

initialize_vars = tf.global_variables_initializer()
saver = tf.train.Saver()

sess.run(initialize_vars)

# insert patterns here to train

# Checkpoint save weights

# model.fit(x=[c1,c2,c3,c4,c5],y=[UV],batch_size=1,epochs=10,verbose=2,callbacks=[checkpoint])
sess.run([loss,optimizer],feed_dict={C1:c1,C2:c2,C3:c3,C4:c4,C5:c5,Y1:UV})
saver.save(sess, "F:\ColourNet Output/modelWeights.ckpt")

