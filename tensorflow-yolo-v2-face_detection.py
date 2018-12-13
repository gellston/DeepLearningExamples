import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

from models.model_yolo_v2_detection import model_yolo_v2_detection


sess = tf.Session()
model = model_yolo_v2_detection(sess=sess, name="model_custom_mobile_detection", num_anchor=5, num_classes=1, learning_rate=1e-3)
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

cost_graph = []
accuracy_graph = []

print('Learning start.')
