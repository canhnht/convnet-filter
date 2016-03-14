from PIL import Image

from keras.preprocessing.image import load_img


def filter_zoom(img, **kwargs):
    zoom_type = kwargs.get('zoom_type', Image.LANCZOS)
    zoom_coef = kwargs.get('zoom_coef', 2)
    org_dim = (img.width, img.height)
    img = img.resize((int(img.width / zoom_coef), int(img.height / zoom_coef)), zoom_type)
    img = img.resize(org_dim, zoom_type)
    return img


def filter_jpg(img, **kwargs):
    img.save('temp.jpg', 'JPEG', quality=40)
    img = load_img('temp.jpg')
    return img


filters = dict(zoom=filter_zoom, jpg=filter_jpg)
