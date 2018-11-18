import tensorflow as tf
import cv2 as cv2
import numpy as np


sess = tf.Session()
saver = tf.train.import_meta_graph('./pretrained-models/mobile_person_segmentation/mobile_person_segmentation.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./pretrained-models/mobile_person_segmentation'))

graph = tf.get_default_graph()
input = graph.get_tensor_by_name("model_custom_mobile_segmentation_v3/input:0")
output = graph.get_tensor_by_name("model_custom_mobile_segmentation_v3/Sigmoid:0")
phase = graph.get_tensor_by_name("model_custom_mobile_segmentation_v3/phase:0")

image = []
npImage = []

cvImage = cv2.imread('D://portrait-dataset//train_input256x256(augmentation)//00001.jpg')
image.append(cvImage)


temp = np.array(image[0])
temp = temp.flatten().reshape([256*256*3])
npImage.append(temp)


feed_dict = {input: npImage, phase: True}
prediction = sess.run(output, feed_dict)


cv2.imshow('input', cvImage)
cv2.imshow('prediection', prediction[0])

cv2.waitKey(0)
