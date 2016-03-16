
# Better image scaling with convolutional neural network

A convolutional neural network tested and trained in scaling up low-resolution images.

# About

This is a trained and tested convolutional neural network based on Keras and Theano. Its purpose is to resolve a bad quality issue when scaling up a small, low-resolution image by big percentage. The network was trained on couple thousand images with approximately 5,000 images per epoch.The image download was also automatized for better efficiency.

# Network Architecture

The convolution layer has 150 9 x 9 filters with a 200 x 200 sized images being the input. After that comes the activation layer (RELU) followed by output layer. The used optimizer is Adam on default parameters. 


# Setup conda environment:

  * create your conda environment
  * install requirements from requirements.txt

# Image Dowload:
  * download and unpack file http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
  * run python image_download.py

In case you did not use Theano framework, amend and put .theanorc into your home directory

# Network Learning:
  * python deep_filters/process-zoom2.py
  
# Testing:
  * python deep_filters/process-zoom2.py --file image_to_enlarge.png
