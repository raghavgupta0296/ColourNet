import vggNet
import imManipulation
import tensorflow as tf
import numpy as np
import scipy.misc as im
import os
import GuassianBlur
import gc

files = os.listdir("./dataImages/dataset")
os.makedirs("./output/tmp")
os.makedirs("./output/ims")
index = 0

def chooose_ims(batch_size):
    global index
    I1 = np.ndarray(shape=[1,256,256,1])
    U1 = np.ndarray(shape=[1,256,256,1])
    V1 = np.ndarray(shape=[1,256,256,1])
    for i in range(batch_size):
        if index>=len(files):
            index=0
        image = im.imread("./dataImages/dataset/"+files[index],mode='RGB')
        image = image/255
        I, U, V = imManipulation.rgb2yuv(image)
        I = np.reshape(I, (1,256, 256,1))
        U = np.reshape(U, (1,256, 256,1))
        V = np.reshape(V, (1,256, 256,1))
        I1 = np.concatenate((I1,I),axis=0)
        U1 = np.concatenate((U1,U),axis=0)
        V1 = np.concatenate((V1,V),axis=0)
        index+=1
    I1 = I1[1:,:,:,:]
    U1 = U1[1:,:,:,:]
    V1 = V1[1:,:,:,:]
    return I1,U1,V1

batch_size = 3

tf.reset_default_graph()

sess = tf.Session()

is_training = tf.placeholder(tf.bool)
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
# c3_ = tf.image.resize_images(C3,(256,256))
# c4_ = tf.image.resize_images(C4,(256,256))
# c5_ = tf.image.resize_images(C5,(256,256))
Wup3 = ini_wt([3,3,128,128])
c3_ = tf.nn.conv2d_transpose(C3,Wup3,[batch_size,256,256,128],strides=[1,2,2,1],padding='SAME')
Wup4 = ini_wt([3,3,256,256])
c4_ = tf.nn.conv2d_transpose(C4,Wup4,[batch_size,256,256,256],strides=[1,4,4,1],padding='SAME')
Wup5 = ini_wt([3,3,512,512])
c5_ = tf.nn.conv2d_transpose(C5,Wup5,[batch_size,256,256,512],strides=[1,8,8,1],padding='SAME')
hypercolumn = tf.concat([C1,C2,c3_,c4_,c5_],3)
Ws1 = ini_wt([3,3,963,128])
y = tf.nn.conv2d(hypercolumn,Ws1,strides=[1,1,1,1],padding='SAME')
y = tf.contrib.layers.batch_norm(y, center=True, scale=True, is_training=is_training,scope='bn1')
y = tf.nn.relu(y)
Ws2 = ini_wt([3,3,128,64])
y = tf.nn.conv2d(y,Ws2,strides=[1,1,1,1],padding='SAME')
y = tf.contrib.layers.batch_norm(y, center=True, scale=True, is_training=is_training,scope='bn2')
y = tf.nn.relu(y)
Ws3 = ini_wt([3,3,64,2])
y = tf.nn.conv2d(y,Ws3,strides=[1,1,1,1],padding='SAME')

def guassianBlurLoss(y,Y1):
    y_u = tf.expand_dims(y[:,:,:,0],3)
    Y1_U = tf.expand_dims(Y1[:,:,:,0],3)
    y_v = tf.expand_dims(y[:,:,:,1],3)
    Y1_V = tf.expand_dims(Y1[:,:,:,1],3)

    u1_blur3 = GuassianBlur.guassianBlur(y_u,3)
    u2_blur3 = GuassianBlur.guassianBlur(Y1_U,3)
    v1_blur3 = GuassianBlur.guassianBlur(y_v,3)
    v2_blur3 = GuassianBlur.guassianBlur(Y1_V,3)

    uv1 = tf.concat((u1_blur3,v1_blur3),3)
    UV1 = tf.concat((u2_blur3, v2_blur3),3)

    u1_blur5 = GuassianBlur.guassianBlur(y_u,5)
    u2_blur5 = GuassianBlur.guassianBlur(Y1_U,5)
    v1_blur5 = GuassianBlur.guassianBlur(y_v,5)
    v2_blur5 = GuassianBlur.guassianBlur(Y1_V,5)

    uv2 = tf.concat((u1_blur5,v1_blur5),3)
    UV2 = tf.concat((u2_blur5, v2_blur5),3)

    l = (tf.reduce_sum(tf.squared_difference(y,Y1))+tf.reduce_sum(tf.squared_difference(uv1,UV1))+tf.reduce_sum(tf.squared_difference(uv2,UV2)))/3
    return l

loss = guassianBlurLoss(y,Y1)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer().minimize(loss)

initialize_vars = tf.global_variables_initializer()
saver = tf.train.Saver()

fno = 1

sess.run(initialize_vars)
vgg = vggNet.vgg16Net(X, './vggWeights/vgg16_weights.npz', sess)

saver.restore(sess, "./model.ckpt")

def make_batch(sess,X,I_,U_,V_):
    I_ = np.concatenate((I_, I_, I_), axis=3)  # 1091,256,256,3
    UV_ = np.concatenate((U_, V_), axis=3)
    c1, c2, c3, c4, c5 = vgg.return_layers(I_,sess)
    return c1,c2,c3,c4,c5,UV_

for n_epochs in range(2000):
    for i in range(0,(int(len(files)/batch_size) - 1),batch_size):

        I,U,V = chooose_ims(batch_size)
        c1,c2,c3,c4,c5,UV = make_batch(sess,X,I,U,V)

        loss_val, op_val, uv_val = sess.run([loss, optimizer, y], feed_dict={C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, Y1: UV, is_training:True})

        print(" loss : ", loss_val)

    # if n_epochs%5 == 0:
        saver.save(sess,"./output/tmp/model.ckpt")
        print(" Model Saved ")
        # For Test
        # image1 = im.imread("unnamed.jpg",mode='RGB')
        # image1 = im.imresize(image1,(256,256,3))
        # image1 = image1/255
        # I1,U1,V1 = imManipulation.rgb2yuv(image1)
        # I1 = np.reshape(I1, (1, 256, 256, 1))
        # U1 = np.reshape(U1,(1,256,256,1))
        # V1 = np.reshape(V1,(1,256,256,1))
        # c1,c2,c3,c4,c5,UV = make_batch(sess,X,I1,U1,V1)
        # loss_val, uv_val = sess.run([loss, y], feed_dict={C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, Y1: UV})
        # u_val1 = uv_val[0, :, :, 0]
        # v_val1 = uv_val[0, :, :, 1]
        # I1 = np.reshape(I1[0], (256, 256))
        # r1, g1, b1 = imManipulation.yuv2rgb(I1, u_val1, v_val1)
        # r1 = np.reshape(r1, (256, 256, 1))
        # g1 = np.reshape(g1, (256, 256, 1))
        # b1 = np.reshape(b1, (256, 256, 1))
        # imp1 = np.concatenate((r1, g1, b1), axis=2)
        # imp1 = imp1 * 255
        # print(imp1, end=" ")
        # im.imsave("./output/ims/colouredImage%d.jpg" % fno, imp1)
        # fno += 1
        image1 = im.imread("unnamed.jpg", mode='RGB')
        image2 = im.imread("unnamed2.jpg", mode='RGB')
        image3 = im.imread("unnamed3.jpg", mode='RGB')
        image1 = im.imresize(image1, (256, 256, 3))
        image2 = im.imresize(image2, (256, 256, 3))
        image3 = im.imresize(image3, (256, 256, 3))
        image1 = image1 / 255
        image2 = image2 / 255
        image3 = image3 / 255
        I1, U1, V1 = imManipulation.rgb2yuv(image1)
        I2, U2, V2 = imManipulation.rgb2yuv(image2)
        I3, U3, V3 = imManipulation.rgb2yuv(image3)
        I1 = np.reshape(I1, (1, 256, 256, 1))
        I2 = np.reshape(I2, (1, 256, 256, 1))
        I3 = np.reshape(I3, (1, 256, 256, 1))
        U1 = np.reshape(U1, (1, 256, 256, 1))
        U2 = np.reshape(U2, (1, 256, 256, 1))
        U3 = np.reshape(U3, (1, 256, 256, 1))
        V1 = np.reshape(V1, (1, 256, 256, 1))
        V2 = np.reshape(V2, (1, 256, 256, 1))
        V3 = np.reshape(V3, (1, 256, 256, 1))
        I0 = np.concatenate((I1, I2, I3), axis=0)
        U0 = np.concatenate((U1, U2, U3), axis=0)
        V0 = np.concatenate((V1, V2, V3), axis=0)
        c1, c2, c3, c4, c5, UV = make_batch(sess, X, I0, U0, V0)
        loss_val, uv_val = sess.run([loss, y], feed_dict={C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, Y1: UV, is_training:False})
        u_val1 = uv_val[0, :, :, 0]
        v_val1 = uv_val[0, :, :, 1]
        I1 = np.reshape(I1[0], (256, 256))
        r1, g1, b1 = imManipulation.yuv2rgb(I1, u_val1, v_val1)
        r1 = np.reshape(r1, (256, 256, 1))
        g1 = np.reshape(g1, (256, 256, 1))
        b1 = np.reshape(b1, (256, 256, 1))
        imp1 = np.concatenate((r1, g1, b1), axis=2)
        imp1 = imp1 * 255
        print(imp1, end=" ")
        im.imsave("./output/ims/colouredImage%d.jpg" % fno, imp1)
        fno += 1
        u_val2 = uv_val[1, :, :, 0]
        v_val2 = uv_val[1, :, :, 1]
        I2 = np.reshape(I2[0], (256, 256))
        r2, g2, b2 = imManipulation.yuv2rgb(I2, u_val2, v_val2)
        r2 = np.reshape(r2, (256, 256, 1))
        g2 = np.reshape(g2, (256, 256, 1))
        b2 = np.reshape(b2, (256, 256, 1))
        imp2 = np.concatenate((r2, g2, b2), axis=2)
        imp2 = imp2 * 255
        print(imp2, end=" ")
        im.imsave("./output/ims/colouredImage%d.jpg" % fno, imp2)
        fno += 1
        u_val3 = uv_val[2, :, :, 0]
        v_val3 = uv_val[2, :, :, 1]
        I3 = np.reshape(I3[0], (256, 256))
        r3, g3, b3 = imManipulation.yuv2rgb(I3, u_val3, v_val3)
        r3 = np.reshape(r3, (256, 256, 1))
        g3 = np.reshape(g3, (256, 256, 1))
        b3 = np.reshape(b3, (256, 256, 1))
        imp3 = np.concatenate((r3, g3, b3), axis=2)
        imp3 = imp3 * 255
        print(imp3, end=" ")
        im.imsave("./output/ims/colouredImage%d.jpg" % fno, imp3)
        fno += 1
