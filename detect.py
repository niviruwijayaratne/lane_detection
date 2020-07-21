import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from mrcnn.config import Config 
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as plt 
import cv2
import numpy as np

# def draw(filename, boxes_list):
#     image = cv2.imread(filename)

#     for box in boxes_list:
#         y1, x1, y2, x2 = box 
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 5)

#     cv2.imshow("image", image)
#     cv2.waitKey(0)

# class TestConfig(Config):
#     NAME = "test"
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = 1 + 80


# rcnn = MaskRCNN(mode="inference", model_dir='./', config=TestConfig())
# rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

# img = load_img('elephant.jpg')
# img = img_to_array(img)

# results = rcnn.detect([img], verbose = 0)
# draw('elephant.jpg', results[0]['rois'])
from mrcnn.visualize import display_instances

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80


def detect(filename):
     # print("SKDJHFLSKDJFHLKSDJHFLSDKJFH")
# define 81 classes that the coco model knowns about
     class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']
     
     # define the test configuration
     # define the model
     rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
     # load coco model weights
     rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
     # load photograph
     img = load_img(filename)
     img = img_to_array(img)
     # make prediction
     results = rcnn.detect([img], verbose=0)
     # get dictionary for first prediction

     r = results[0]
     bounds = r['rois']
     im = cv2.imread(filename)
     for i in range(0, len(bounds)):
          arr = bounds[i]
          cv2.rectangle(im, (arr[1], arr[0]), (arr[3], arr[2]), (0, 255,0), 2)


     # print("BOUNDS_________________________", bounds)
     
     # cv2.rectangle(im, (bounds[1], bounds[0]), (bounds[3], bounds[2]), (255,0,0),  5)
     # cv2.imshow('bounded',im)
     # cv2.waitKey(0)
     return im