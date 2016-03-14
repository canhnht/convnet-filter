from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam


def get_model_zoom(feature_size=128, image_size=128, filter_size=8, channels=3):
    model = Sequential()
    model.add(Convolution2D(feature_size, filter_size, filter_size, input_shape=(channels, image_size, image_size),
                            border_mode='valid', activation='relu'))
    model.add(Convolution2D(channels, 1, 1, border_mode='valid'))

    optimizer = Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
