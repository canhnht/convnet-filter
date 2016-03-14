import argparse
import logging
from logging.config import fileConfig

from PIL import Image

from deep_filters.core.models import get_model_zoom
from deep_filters.core.util import predict_base_image, process_learning
from keras.preprocessing.image import load_img

MODEL_NAME = 'zoom2'

fileConfig('logging.conf')
logger = logging.getLogger('nn')


FEATURE_SIZE = 350
CONV_SIZE = 8+1
CHANNELS = 1
ZOOM_LEARN = 2
ZOOM_TYPE = Image.LANCZOS
MODE = "YCbCr"    #YCbCr / RGB
IMAGE_SIZE = 200


def train_on_image_net():
    logger.debug('Learning mode.')
    logger.debug('Initializing model')
    model = get_model_zoom(FEATURE_SIZE, IMAGE_SIZE, CONV_SIZE, CHANNELS)
    img_filter = 'zoom'
    test_images = 'test_images'

    process_learning(img_filter, model, MODEL_NAME, MODE, CHANNELS, zoom_learn=ZOOM_LEARN, test_images_path=test_images)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="file to process")
    parser.add_argument("--zoom", help="repeat zoom", type=int, default=1)
    args = parser.parse_args()

    if not args.file:
        return train_on_image_net()

    logger.debug('Start processing')
    logger.debug('Loading image')

    img = load_img(args.file)

    logger.debug('Opening zoom model')
    model = get_model_zoom(FEATURE_SIZE, IMAGE_SIZE, CONV_SIZE, CHANNELS)
    logger.debug('Loading zoom model')
    model.load_weights('weights/' + MODEL_NAME)

    logger.debug('Resizing image with standard filter.')
    img = img.resize((int(img.width * ZOOM_LEARN), int(img.height * ZOOM_LEARN)), ZOOM_TYPE)
    logger.debug('Zooming.')
    img = predict_base_image(CHANNELS, img, model, MODE)

    img.save('result.png', 'PNG')
    img.show()

if __name__ == '__main__':
    run()
