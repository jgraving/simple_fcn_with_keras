import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import glob
import os

def binarylab(labels,nb_class,size):
    y = np.zeros((size, size,nb_class))
    for i in range(size):
        for j in range(size):
            y[i, j,labels[i][j]] = 1
    return y

def load_image(path, size):
    img = Image.open(path)
    img = img.resize((size, size))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X = preprocess_input(X)
    return X

def load_label(path, size, nb_class):
    label = Image.open(path)
    label = label.resize((size, size))
    y = np.array(label, dtype=np.int32)
    mask = y == 255
    y[mask] = 0
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
