import numpy as np
from PIL import Image
from os import path

pascal_palette = np.array([(0, 0, 0)
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)], dtype=np.uint8)
output_dir = "./Result/"
def save_seg(result, name):
    '''
    convert prediction result to PNG image and save
    :param result: fcn prediction result
    :param name: image name for saving
    :return:
    '''
    prediction = np.argmax(result, axis=-1)

    # Apply the color palette to the segmented image
    color_image = np.array(pascal_palette)[prediction.ravel()].reshape(prediction.shape + (3,))
    Image.fromarray(color_image).save(path.join(output_dir,name),"png")
    return
