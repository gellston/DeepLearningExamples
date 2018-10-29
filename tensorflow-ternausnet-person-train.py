import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

from models.model_ternausnet import model_ternausnet
from pycocotools.coco import COCO


sess = tf.Session()
model1 = model_ternausnet(sess=sess, name="person_ternausnet", learning_rate=0.0015)

sess.run(tf.global_variables_initializer())

dataDir='D://coco-dataset'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds);

for index in range(len(imgIds)):
    img = coco.loadImgs(imgIds[index])
    file_name = img[0]['file_name']
    local_image = cv2.imread(('D://coco-dataset//val2017//'+ file_name))
    height, width, channels = local_image.shape

    annIds = coco.getAnnIds(imgIds=img[0]['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    #coco.showAnns(anns)

    mask = np.zeros([height, width], dtype=np.uint8)
    for ann in anns:
        mask += coco.annToMask(ann) * 255

    x_data = []
    y_data = []

    temp = np.array(cv2.resize(local_image, (256, 256)))
    temp = temp.flatten().reshape([256 * 256 * 3])
    x_data.append(temp)

    temp = np.array(cv2.resize(mask, (256, 256)))
    temp = temp.flatten().reshape([256 * 256])
    y_data.append(temp)

    cost, _ = model1.train(x_data, y_data, keep_prop=True)
    print('cost =', '{:.9f}'.format(cost), '\n');



    cv2.imshow('validation', local_image)
    cv2.imshow("mask", mask)
    cv2.waitKey(10)


