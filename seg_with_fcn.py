from keras.layers import Convolution2D, MaxPooling2D,Input,Activation,UpSampling2D,Permute,Reshape,merge
from keras.models import Model
from preprocess import *
from keras.optimizers import Adam
from save_segment import *

def build_segmentor(img_h=224,img_w=224):
    FCN_CLASSES = 21
    # (samples, channels, rows, cols)
    input_img = Input(shape=[img_h,img_w,3])
    # (3*224*224)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # (64*112*112)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # (128*56*56)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    p3 = x
    p3 = Convolution2D(FCN_CLASSES, 1, 1,activation='relu',border_mode='valid')(p3)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(p3)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    p4 = x
    p4 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p4)
    p4 = UpSampling2D((2, 2))(p4)
    p4 = Convolution2D(FCN_CLASSES, 3, 3, activation='relu', border_mode='same')(p4)

    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    p5 = x
    p5 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p5)
    p5 = UpSampling2D((4, 4))(p5)
    p5 = Convolution2D(FCN_CLASSES, 3, 3, activation='relu', border_mode='same')(p5)

    h = merge([p3, p4, p5], mode='sum')
    h = UpSampling2D((8, 8))(h)
    h = Convolution2D(FCN_CLASSES, 3, 3, activation='relu', border_mode='same')(h)
    h = Reshape([img_h*img_w,FCN_CLASSES])(h)
    h = Permute([2,1])(h)
    h = Activation("softmax")(h)
    h = Permute([2,1])(h)
    out = Reshape([img_h,img_w,FCN_CLASSES])(h)

    model = Model(input_img, out)
    return model

if __name__ == '__main__':
    size = 224
    (X_train, y_train), (X_test, y_test) = load_dataset(nb_iamges=10,bk_point=8)
    print X_train.shape
    print X_test.shape
    print y_train.shape
    print y_test.shape

    opt = Adam
    segmentor = build_segmentor()
    segmentor.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    segmentor.fit(X_train,y_train,batch_size=32,nb_epoch=50,verbose=1,validation_data=(X_test,y_test),shuffle=True)

    pred = segmentor.predict(X_test)
    print pred.shape
    print pred[0].shape
    for i in range(pred.shape[0]):
        save_seg(pred[i],"out{}".format(i))
