import imgaug.augmenters as iaa
import random
import cv2
import numpy as np
from PIL import Image
import time
from skimage.transform import resize

def init_random_params():
    pass

def pad_to_size(image, width, height):
    pad_to_size = iaa.PadToFixedSize(width=width, height=height)
    return pad_to_size(image = image)

def resize(image, height):
    resize = iaa.Resize({"height": height, "width": "keep-aspect-ratio"})
    return resize(image = image)

def augment(image):
    orig = image
    print("TRYING AUG")
    # percentage of image, size of dropouts, channel
    #coarse_dropout = iaa.CoarseDropout(0.5, size_percent=0.8, per_channel=0.5)
    pad_to_size = iaa.PadToFixedSize(width=400, height=400)
    image_aug = pad_to_size(image = image)
    # top, right, bottom, left
    #val = (189,190)
    #crop_pad = iaa.CropAndPad(percent=(-0.25, 0.25))
    #image_aug = crop_pad(image_aug)


    print("aug shape ", image_aug.shape)
    #print("orig shaoe ", orig.shape)

    return image_aug


if __name__ == '__main__':
    PATH = "/home/sally/Work/Promotion/Data/sample.tif"
    im = np.array(Image.open(PATH))
    print(im.shape)
    shape = (400, 400)

    #(1000, 4096)
    y,x = im.shape
    cropx_t = 1000
    cropy_t = 0000
    cropx_b = 2800
    cropy_b = 1000
   
    img = im[cropy_t:cropy_b,cropx_t:cropx_b]

    img = Image.fromarray(img)
    img.thumbnail(shape, Image.ANTIALIAS)

    img = np.array(img)
    orig = img.astype(np.uint8)
    res = Image.fromarray(orig)
    res.show()
    time.sleep(10)
    """
    aug_img = augment(orig)

    numpy_horizontal =  np.concatenate((orig, aug_img), axis=1)
    res = Image.fromarray(numpy_horizontal)
    res.show()
    time.sleep(10)"""


