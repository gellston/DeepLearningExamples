import tensorflow as tf
import datasetloader as loader
import cv2 as cv2
import numpy as np



sess = tf.Session()
saver = tf.train.import_meta_graph('./digits-trained-model/digits_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./digits-trained-model'))


graph = tf.get_default_graph()
output = graph.get_tensor_by_name("output:0")
input = graph.get_tensor_by_name("input:0")

image = []
npImage = []
image.append(cv2.imread('./digits_validation/0_zero/zero_2.jpg'))
image.append(cv2.imread('./digits_validation/3_three/three_5.jpg'))
image.append(cv2.imread('./digits_validation/7_seven/seven_6.jpg'))

for index in range(3):
    temp = np.array(image[index])
    temp = temp.flatten().reshape([28*28*3])
    npImage.append(temp)

feed_dict = {input: npImage}

prediction = sess.run(output, feed_dict)

print ('=== prediction values ===')
print(prediction[0], '\n')
print(prediction[1], '\n')
print(prediction[2], '\n')


cv2.imshow('image' + str(0), image[0])
cv2.imshow('image' + str(3), image[1])
cv2.imshow('image' + str(7), image[2])
cv2.waitKey(0)
