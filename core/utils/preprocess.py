__author__ = 'yunbo'

import numpy as np
from PIL import Image
import time

DEBUG = True

def reshape_patch(img_tensor, patch_size):

    """if DEBUG:
        print("shape ", img_tensor.shape)
        test = img_tensor
        test = test.transpose(0,4,1,3,2)
        print("new shape 1 ", test.shape)
        test = np.reshape(test, (4,test.shape[3],test.shape[4]))
        print("new shape 2 ", test.shape)
        X1, X2, X3, X4 = np.split(test, 4, axis=0)
        X1 = np.reshape(X1, (test.shape[1],test.shape[2]))
        print("final shape ", X1.shape)


        img = Image.fromarray(X1.astype(np.uint8), 'L')
        img.show(title="input")
        time.sleep(10)
    """

    assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size*patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    
    if DEBUG:
        print("shape ", img_tensor.shape)
        test = img_tensor
        test = test.transpose(0,4,1,3,2)
        print("new shape 1 ", test.shape)
        test = np.reshape(test, (4,test.shape[3],test.shape[4]))
        print("new shape 2 ", test.shape)
        X1, X2, X3, X4 = np.split(test, 4, axis=0)
        X1 = np.reshape(X1, (test.shape[1],test.shape[2]))
        print("final shape ", X1.shape)


        img = Image.fromarray(X1.astype(np.uint8), 'L')
        img.show(title="output")
        time.sleep(10)

    return img_tensor

