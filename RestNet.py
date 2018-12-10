import numpy as np
from resnet50 import ResNet50
from keras.preprocessing import image
from keras_applications import resnet50

def main():
    ResNet50_model()

def ResNet50_model():
    model = resnet50()