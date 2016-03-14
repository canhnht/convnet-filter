from logging.config import fileConfig, logging
from urllib.request import urlopen

from PIL import Image
from six import BytesIO

fileConfig('logging.conf')
logger = logging.getLogger('nn')

def download():
    image_set_path = './fall11_urls.txt'
    image_number = 0
    with open(image_set_path, "r") as ins:

        for line in ins:
            try:
                image = line.split()[1]
                print("Loading {} image from {}".format(image_number, image))
                image_response = urlopen(image, timeout=1)
                if image_response.url.endswith('photo_unavailable.png'):
                    continue
                img = Image.open(BytesIO(image_response.read()))

                img.save('imgnet/image-{}.png'.format(image_number), 'PNG')
                image_number += 1
            except:
                logger.exception("Error")
            if image_number > 5000:
                break

if __name__ == "__main__":
    download()
