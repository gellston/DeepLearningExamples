import tensorflow as tf
import datasetloader as loader
import cv2 as cv2
import numpy as np



sess = tf.Session()
saver = tf.train.import_meta_graph('./pre-trained-model/digits.model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./pre-trained-model'))


graph = tf.get_default_graph()
Output = graph.get_tensor_by_name("output:0")
Input = graph.get_tensor_by_name("Input:0")

image = []
npImage = []
image.append(cv2.imread('./digits/0_zero/1_1_0_zero.jpg'))
image.append(cv2.imread('./digits/3_three/1_1_3_three.jpg'))
image.append(cv2.imread('./digits/7_seven/1_1_7_seven.jpg'))

for index in range(3):
    temp = np.array(image[index])
    temp = temp.flatten().reshape([28*28*3])
    npImage.append(temp)

feed_dict = {Input: npImage}

prediction = sess.run(Output, feed_dict)

print ('=== prediction values ===')
print(prediction[0], '\n')
print(prediction[1], '\n')
print(prediction[2], '\n')


cv2.imshow('image' + str(0), image[0])
cv2.imshow('image' + str(3), image[1])
cv2.imshow('image' + str(7), image[2])
cv2.waitKey(0)
