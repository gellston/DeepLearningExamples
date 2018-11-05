import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from pycocotools.coco import COCO

dataDir='D://coco-dataset'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person']);
imageIDs = coco.getImgIds(catIds=catIds);

for index in range(len(imageIDs)):
    images = coco.loadImgs(imageIDs[index])
    image_id = images[0]['id']
    annIds = coco.getAnnIds(imgIds=image_id, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    is_area_small = False

    for ann in anns:
        if ann['area'] < 30000:
            is_area_small = True

    for ann in anns:
        if ann['area'] > 70000:
            is_area_small = True

    if is_area_small == True:
        continue

    file_name = images[0]['file_name']
    local_image = cv2.imread(('D://coco-dataset//train2017//' + file_name))
    height, width, channels = local_image.shape
    mask = np.zeros([height, width], dtype=np.uint8)
    for ann in anns:
        print('area =', '{:.9f}'.format(ann['area']), '\n');
        mask += coco.annToMask(ann) * 255

    resize_local_image = cv2.resize(local_image, (256, 256))
    resize_mask_image = cv2.resize(mask, (256, 256))

    cv2.imwrite('D://coco-dataset//coco-mobile-v2//train-input//' + file_name, resize_local_image)
    cv2.imwrite('D://coco-dataset//coco-mobile-v2//train-label//' + file_name, resize_mask_image)

    cv2.imshow('validation', resize_local_image)
    cv2.imshow("mask", resize_mask_image)
    cv2.waitKey(1)


