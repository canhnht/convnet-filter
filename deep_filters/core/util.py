import logging
import os
import random
import shelve
import time
from logging.config import fileConfig

import numpy as np
from PIL import Image
from tqdm import *

from deep_filters.core.images import load_array_image, process_image
from deep_filters.core.keras_callbacks import ModelBestCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import load_img, img_to_array

fileConfig('logging.conf')
logger = logging.getLogger('nn')


def predict_base_image(channels, img, model, mode, shape=None):
    return predict_base_image_channels_1(channels, img, model, mode, shape=shape)


def predict_base_image_channels_1(channels, img, model, mode, shape=None):
    wx = model.layers[-1].output_shape[2]
    wy = model.layers[-1].output_shape[3]

    logger.debug('Processing with neural model. Image size: {} {}'.format(wx, wy))
    if mode == 'YCbCr':
        img = img.convert('YCbCr')
    img_ar = np.asarray(img, dtype='float32')
    img_ar = img_ar.transpose(2, 1, 0)

    full_time = 0
    for y in tqdm(range(0, img.height, wy)):
        for x in range(0, img.width, wx):

            valid_x = model.layers[0].input_shape[2]
            if x + valid_x > img.width:
                valid_x = img.width - x
            valid_y = model.layers[0].input_shape[3]
            if y + valid_y > img.height:
                valid_y = img.height - y

            valid_x2 = wx
            if x + valid_x2 > img.width:
                valid_x2 = img.width - x
            valid_y2 = wy
            if y + valid_y2 > img.height:
                valid_y2 = img.height - y

            if channels == 3:
                cropped_input = np.zeros((channels, model.layers[0].input_shape[2], model.layers[0].input_shape[3]), dtype='float32')
                cropped_input[:, :valid_x, :valid_y] = img_ar[:, x:x+valid_x, y:y+valid_y]
                start_time = time.process_time()
                preds = model.predict(np.array([cropped_input]))

                full_time += (time.process_time() - start_time)
                preds = np.clip(preds, 0, 255)
                img_ar[:, x:x+valid_x2, y:y+valid_y2] = preds[0][:, :valid_x, :valid_y]

            else:
                for c in range(0, 1 if mode == 'YCbCr' else 3):
                    cropped_input = np.zeros((1, model.layers[0].input_shape[2], model.layers[0].input_shape[3]), dtype='float32')
                    cropped_input[0, :valid_x, :valid_y] = img_ar[c, x:x+valid_x, y:y+valid_y]
                    start_time = time.process_time()
                    if mode == 'YCbCr':
                        preds = model.predict(cropped_input.reshape((1, 1, cropped_input.shape[1], cropped_input.shape[2])))
                    else:
                        p = cropped_input[0]
                        preds = model.predict(p.reshape((1, 1, p.shape[0], p.shape[1])))

                    full_time += (time.process_time() - start_time)
                    preds = np.clip(preds, 0, 255)
                    img_ar[c, x:x+valid_x2, y:y+valid_y2] = preds[0][0, :valid_x, :valid_y]

    if mode == 'YCbCr':
        result = img_ar.transpose(2, 1, 0).astype("uint8")
        result = Image.fromarray(result[:, :, :], "YCbCr")
        result = result.convert("RGB")
    else:
        img_ar = img_ar.transpose(2, 1, 0)
        result = Image.fromarray(img_ar.astype("uint8"), "RGB")
    logger.debug('End of processing, nn time: {}'.format(full_time))
    return result


def get_size(l):
    sum = 0
    for x in l:
        sum += x.size
    return sum


def get_images(image_path):
    for dirName, subdirList, fileList in os.walk(image_path):
        return [os.path.join(image_path, f) for f in fileList]


def process_learning(img_filter, model, model_name, mode, channels, image_path='imgnet', monitor='val_loss',
                     test_images_path=None, samples=None, epochs=10, shape=None, resize=False, **kwargs):
    try:
        logger.debug('Loading weights')
        model.load_weights('weights/' + model_name)
    except:
        logger.exception('Weights not found, learning from scratch.')

    logger.debug("Model layers:")
    for layer in model.layers:
        logger.debug(layer.output_shape)

    if not os.path.exists('weights'):
        os.makedirs('weights')
    model_dict = shelve.open('weights/' + model_name + '_storage')
    progress = model_dict.get('progress', 0)
    if not shape:
        shape = (model.layers[-1].output_shape[2], model.layers[-1].output_shape[3])

    test_images = get_images(test_images_path)
    logger.debug(test_images)

    logger.debug("Kernel size: {}".format(shape))
    X_test, y_test = load_array_image(test_images, mode=mode, kernel=shape, img_filter=img_filter, channels=channels, model=model, resize=resize, **kwargs)
    image_number = 0
    logger.debug('Skipping {} lines'.format(progress))
    logger.debug('Current val_loss {}'.format(model_dict.get('best', np.Inf)))
    random.seed()

    l = []
    t = []

    save_best = True if monitor == 'val_loss' else False
    check_point = ModelBestCheckpoint('weights/' + model_name, monitor='val_loss', verbose=1, save_best_only=save_best, best=model_dict.get('best', np.Inf))
    samples_size = 50000000

    for x in range(0, 100):
        for dirName, subdirList, fileList in os.walk(image_path):
            # with open(image_set_path, "r") as ins:

            for image_response in fileList:
                for _ in range(progress):
                    continue
                try:
                    image_number += 1
                    image = image_path + '/' + image_response
                    img = load_img(image)
                    if resize:
                        img = img.resize(shape)

                    process_image(img, shape, l, t, mode=mode, omit_coef=0.1, img_filter=img_filter, channels=channels, model=model, **kwargs)

                    if get_size(l) > samples_size:
                        X_train = np.array(l)
                        Y_train = np.array(t)
                        early_stopping = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=1)
                        callbacks = [check_point]
                        if save_best:
                            callbacks.append(early_stopping)
                        model.fit(X_train, Y_train, verbose=1, batch_size=1, nb_epoch=epochs, validation_data=(X_test, y_test), callbacks=callbacks)
                        l = []
                        t = []
                        logger.debug("Progress: {}. Best loss so far: {}".format(progress, check_point.best))
                        model_dict['best'] = check_point.best
                        model_dict['progress'] = progress
                        image_number = 0

                except Exception as ex:
                    print(ex)
                    pass
                progress += 1
