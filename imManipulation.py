import scipy.misc as im
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def rgb2yuv(i):
    r = i[:,:,0]
    g = i[:,:,1]
    b = i[:,:,2]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = 0.492 * (b - Y)
    V = 0.877 * (r - Y)
    return Y, U, V

def yuv2rgb(Y,U,V):
    r = Y + 1.14 * V
    g = Y - 0.395 * U - 0.581 * V
    b = Y + 2.033 * U
    return r,g,b

if __name__=='__main__':
    i = im.imread("bea1.jpg")

    # i = np.array(i)
    sh = i.shape
    # print (i)
    # plt.imshow(i)
    # plt.show()

    Y,U,V = rgb2yuv(i)
    r,g,b = yuv2rgb(Y,U,V)

    x = np.zeros(sh)
    x[:,:,0] = r
    x[:,:,1] = g
    x[:,:,2] = b
    # x = np.transpose(x,[1,2,0])

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if(x[i][j][k]<0):
                    x[i][j][k]=0
                x[i][j][k] = round(x[i][j][k])

    print ("----------------")

    im.imsave("temp.jpg",x)

    # plt.imshow(x)
    # print (x)
    # plt.show()
