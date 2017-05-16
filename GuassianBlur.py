import numpy as np
import tensorflow as tf
import scipy.misc as im

def guassianBlur(image,kernel_size,mean=0.5,std=1.0):
    l = 1/(kernel_size**2+1)
    weights = []
    for i in range(1,kernel_size**2+1):
        weights.append(l*i)
    weights = np.array(weights)
    weights = 1 / (np.sqrt(2 * 3.14 * std)) * np.exp(-((weights - mean) ** 2 / 2 * std))
    weights = np.reshape(weights,(kernel_size,kernel_size,1,1))
    blur = tf.nn.conv2d(image,weights,strides=[1,1,1,1],padding='SAME')
    blur = (1/kernel_size)*blur
    return blur

if __name__ == '__main__':
    image = im.imread("bea1.jpg",mode='RGB')
    image = np.reshape(image,(1,256,256,3))
    X = tf.placeholder(tf.float32,shape=(1,256,256,3))
    blur = guassianBlur(X,3)
    with tf.Session() as sess:
        blurredImage = sess.run([blur],feed_dict={X:image})
    blurredImage = np.round(blurredImage)
    blurredImage = np.reshape(blurredImage,(256,256,3))
    im.imsave("blurred.jpg",blurredImage)

