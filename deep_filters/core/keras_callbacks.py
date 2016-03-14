'''This provides visualization tools for Keras.'''
import subprocess
import warnings

import numpy as np
from PIL import Image
from bokeh.plotting import (cursession, figure, output_server, show)

from keras.callbacks import Callback
from keras.preprocessing.image import load_img


class BokehCallback(Callback):

    def __init__(self, predit_funct=None):
        Callback.__init__(self)
        # output_notebook()
        self.loss = np.array([])
        self.psnrs = np.array([])
        output_server("line")
        self.imagew = 512
        self.min_loss = 10000
        self.predit_funct = predit_funct
        self.p = figure()
        self.p2 = figure()
        self.x = np.array([])
        self.y = np.array([])
        self.bx = np.array([])
        self.by = np.array([])
        self.cx = np.array([])
        self.epochNo = 0
        self.p.line(self.x, self.y, name='line', color="tomato", line_width=2)
        self.p.line(self.bx, self.by, name='batch_line', color="blue", line_width=2)
        self.p2.line(self.cx, self.psnrs, name='psnr', color="green", line_width=2)
        show(self.p)
        # show(self.p2)
        # self.p2 = figure(x_range=[0, self.imagew], y_range=[0, self.imagew])
        # self.p2.image_rgba(name='image', image=[np.array((self.imagew, self.imagew), dtype='uint32')], x=0, y=0, dw=self.imagew, dh=self.imagew)
        # show(self.p2)
        self.psnr = 0

    def on_batch_end(self, batch, logs={}):
        self.loss = np.append(self.loss, logs['loss'])
        # renderer = self.p.select(dict(name="batch_line"))
        # ds = renderer[0].data_source
        # ds.data['x'] = self.bx
        # ds.data['y'] = self.by
        # ds.push_notebook()


    def on_epoch_end(self, epoch, logs={}):
        epoch = self.epochNo
        self.x = np.append(self.x, epoch)
        self.y = np.append(self.y, logs['val_loss'])
        self.bx = np.append(self.bx, epoch)
        self.by = np.append(self.by, self.loss.mean())
        self.loss = np.array([])
        self.cx = np.append(self.cx, epoch)
        renderer = self.p.select(dict(name="line"))
        ds = renderer[0].data_source
        ds.data['x'] = self.x
        ds.data['y'] = self.y
        cursession().store_objects(ds)
        # ds.push_notebook()
        renderer = self.p.select(dict(name="batch_line"))
        ds = renderer[0].data_source
        ds.data['x'] = self.bx
        ds.data['y'] = self.by
        # ds.push_notebook()
        cursession().store_objects(ds)
        # if logs['val_loss'] < self.min_loss:
        if self.predit_funct:
            self.psnr = self.predit_funct(self.model, epoch)
            print("psnr: {}".format(self.psnr))

            self.psnrs = np.append(self.psnrs, self.psnr)
            renderer = self.p2.select(dict(name="psnr"))
            ds = renderer[0].data_source
            ds.data['x'] = self.x
            ds.data['y'] = self.psnrs
            cursession().store_objects(ds)

        self.min_loss = min(self.min_loss, logs['val_loss'])
        self.epochNo += 1


class ModelPsnrCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, predict_funct=None):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best = -np.Inf
        self.predict_funct = predict_funct

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = self.predict_funct(self.model, epoch)
            print("psnr: {}".format(current))
            if current > self.best:
                if self.verbose > 0:
                    print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                          % (epoch, self.monitor, self.best, current, filepath))
                self.best = current
                self.model.save_weights(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)


class ModelBestCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, best=np.Inf):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best = best

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Can save best model only with %s available, skipping." % (self.monitor), RuntimeWarning)
            else:
                if current < self.best:
                    if self.verbose > 0:
                        print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                              % (epoch, self.monitor, self.best, current, filepath))
                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)


class ModelBestDictCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, model_dict=None):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.model_dict = model_dict
        self.best = model_dict.get('best', np.Inf)
        self.viewer = None
        self.no = 0

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Can save best model only with %s available, skipping." % (self.monitor), RuntimeWarning)
            else:
                if current < self.best:
                    if self.verbose > 0:
                        print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                              % (epoch, self.monitor, self.best, current, filepath))
                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)
                    self.model_dict['best'] = current
                    img = load_img('/home/robin/test-deblur-test.png')
                    img = predict_base_image_channels_1(1, img, self.model, 'RGB')
                    if self.viewer:
                        close(self.viewer)
                    self.viewer = show_img(img, self.no)
                    self.no += 1
                    # img.save('/home/robin/test-deblur.png', 'PNG')
                else:
                    if self.verbose > 0:
                        print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)


def predict_base_image_channels_1(channels, img, model, mode, shape=None):
    wx = model.layers[-1].output_shape[2]
    wy = model.layers[-1].output_shape[3]

    if mode == 'YCbCr':
        img = img.convert('YCbCr')
    img_ar = np.asarray(img, dtype='float32')
    img_ar = img_ar.transpose(2, 1, 0)

    full_time = 0
    for y in range(0, img.height, wy):
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
                preds = model.predict(np.array([cropped_input]))
                preds = np.clip(preds, 0, 255)
                img_ar[:, x:x+valid_x2, y:y+valid_y2] = preds[0][:, :valid_x, :valid_y]

            else:
                for c in range(0, 1 if mode == 'YCbCr' else 3):
                    cropped_input = np.zeros((1, model.layers[0].input_shape[2], model.layers[0].input_shape[3]), dtype='float32')
                    cropped_input[0, :valid_x, :valid_y] = img_ar[c, x:x+valid_x, y:y+valid_y]
                    if mode == 'YCbCr':
                        preds = model.predict(cropped_input.reshape((1, 1, cropped_input.shape[1], cropped_input.shape[2])))
                    else:
                        p = cropped_input[0]
                        preds = model.predict(p.reshape((1, 1, p.shape[0], p.shape[1])))

                    preds = np.clip(preds, 0, 255)
                    img_ar[c, x:x+valid_x2, y:y+valid_y2] = preds[0][0, :valid_x, :valid_y]

    if mode == 'YCbCr':
        result = img_ar.transpose(2, 1, 0).astype("uint8")
        result = Image.fromarray(result[:, :, :], "YCbCr")
        result = result.convert("RGB")
    else:
        # result = array_to_img(img_ar, scale=False)
        img_ar = img_ar.transpose(2, 1, 0)
        result = Image.fromarray(img_ar.astype("uint8"), "RGB")
    return result


def show_img(img, no=0):
    name = '/home/robin/test-blur6-{}.png'.format(str(no))
    img.save(name)
    viewer = subprocess.Popen(['shotwell', name])
    return viewer


def close(viewer):
    viewer.terminate()
    viewer.kill()