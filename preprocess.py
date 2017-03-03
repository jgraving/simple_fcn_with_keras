import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import glob
import os

def binarylab(labels,nb_class,size):
    y = np.zeros((size, size,nb_class))
    for i in range(size):
        for j in range(size):
            y[i, j,labels[i][j]] = 1
    return y

def load_img_array(fname, target_size=None, dim_ordering='default'):
    """Loads and image file and returns an array."""
    img = Image.open(fname)
    img = img.resize(target_size)
    img.load()
    x = image.img_to_array(img, dim_ordering=dim_ordering)
    img.close()
    return x

def load_image(path, size):
    img = Image.open(path)
    img = img.resize((size, size))
    img.load()
    X = np.array(img)
    img.close()

    #centralize
    pascal_mean = np.array([102.93, 111.36, 116.52])
    X = X - pascal_mean
    X = np.expand_dims(X, axis=0)
    return X

def load_label(path, size, nb_class):
    y = load_img_array(path,target_size=[224,224])
    y = np.reshape(y,[224,224])
    mask = y==255
    y[mask]=0
    y = binarylab(y, nb_class, size)  # 21 classes
    y = np.expand_dims(y, axis=0)
    return y

def load_dataset(nb_iamges=100,bk_point=80):
    #configures for images
    size = 224
    nb_class = 21
    load_log = open("./load_log.txt", "w")

    images = np.zeros((1, size, size,3))
    labels = np.zeros((1, size, size,nb_class))
    images_path = "./VOCdevkit/VOC2012/JPEGImages/"
    for file in glob.glob(images_path + "*.jpg"):
        X = load_image(file, size)
        path_y = file.replace("JPEGImages", "SegmentationClass").replace("jpg", "png")
        if os.path.exists(path_y):
            y = load_label(path_y, size, nb_class)
            images = np.append(images, X,axis=0)
            labels = np.append(labels, y, axis=0)
            load_log.write(file+"\n")
        if images.shape[0] > nb_iamges:
            break
    load_log.close()
    X = np.array(images)
    y = np.array(labels)
    nb_X = X.shape[0]
    nb_y = y.shape[0]

    X_train = X[1:bk_point+1,:,:,:]
    X_test = X[bk_point+1:nb_X, :,:,:]
    y_train = y[1:bk_point+1,:,:,:]
    y_test = y[bk_point+1:nb_y,:,:,:]
    return (X_train, y_train), (X_test, y_test)
