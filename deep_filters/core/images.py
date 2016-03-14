import numpy as np

from deep_filters.core.filters import filters
from keras.preprocessing.image import load_img, img_to_array


def load_array_image(paths, mode, kernel=(128, 128), img_filter='zoom', channels=3, model=None, zoom_learn=2, resize=False, **kwargs):
    """
    3 channels as 3 batch datas
    :param path:
    :return:
    """
    l = []
    t = []
    for f in paths:
        print(f)
        img = load_img(f)
        if resize:
            img = img.resize(kernel)
        image_name = f.split('/')[-1]
        process_image(img, kernel, l, t, mode, img_filter=img_filter, channels=channels, model=model, image_name=image_name, **kwargs)
    return np.array(l), np.array(t)


def process_image(img, kernel, l, t, mode, omit_coef=0.5, img_filter='zoom', channels=3, model=None, **kwargs):
    if filters.get(img_filter):
        img_to_process = filters[img_filter](img, **kwargs)
    else:
        img_to_process = load_img(img_filter)
        img_to_process = img_to_process.resize(img.size)
    crop_whole_image(channels, img, img_to_process, kernel, l, mode, model, omit_coef, t)


def crop_whole_image(channels, img, img_to_process, kernel, l, mode, model, omit_coef, t):
    for y in range(0, img.height, int(kernel[1])):
        for x in range(0, img.width, int(kernel[0])):
            crop_image(channels, img, kernel, mode, model, t, x, y, 'OUTPUT')
            crop_image(channels, img_to_process, kernel, mode, model, l, x, y)


def crop_one_image(channels, img, kernel, mode, model, t, learn_mode):
    for y in range(0, img.height, int(kernel[1])):
        for x in range(0, img.width, int(kernel[0])):
            crop_image(channels, img, kernel, mode, model, t, x, y, learn_mode)


def crop_image(channels, img, kernel, mode, model, t, x, y, learn_mode='INPUT'):

    if learn_mode == 'INPUT' and model:
        posx = (model.layers[0].input_shape[2] - kernel[0])
        posy = (model.layers[0].input_shape[3] - kernel[1])
    else:
        posx = 0
        posy = 0

    img_c = img.crop((x, y, x + kernel[0] + posx, y + kernel[1] + posy))
    img_converted = img_c
    if mode == 'YCbCr':
        img_converted = img_c.convert('YCbCr')
        ar = img_to_array(img_converted)
        input_array = ar[0].reshape(1, img_c.width, img_c.height)
        t.append(input_array)
    else:
        if channels == 1:
            ar = img_to_array(img_converted)
            for c in range(0, 3):
                ar_new = ar[c].reshape(1, img_converted.width, img_converted.height)
                t.append(ar_new)
        else:
            ar = img_to_array(img_converted)
            t.append(ar)
    return img_c