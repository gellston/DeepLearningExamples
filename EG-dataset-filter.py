import os
import cv2 as cv2


image_data_path = 'D://portrait-dataset//EG_code_data_release//[EG_code_data]_release//data//images_data_crop//'
mask_data_path = 'D://portrait-dataset//EG_code_data_release//[EG_code_data]_release//data//image_mask_jpg//'

target_image_path = 'D://portrait-dataset//train_input600x800//'
target_mask_path = 'D://portrait-dataset//train_label600x800//'

count = 0
for filename in os.listdir(image_data_path):

    if filename == '.DS_Store': continue

    count +=1
    print(filename)

    source_image = cv2.imread(image_data_path + filename)
    source_mask = cv2.imread(mask_data_path + os.path.splitext(filename)[0] + '_mask.jpg')

    #modified_image = cv2.resize(source_image, (600, 800))
    #modified_mask = cv2.resize(source_mask, (600, 800))
    modified_image = source_image
    modified_mask = source_mask

    cv2.imwrite(target_image_path + filename, modified_image)
    cv2.imwrite(target_mask_path + filename, modified_mask)
    cv2.imshow('source', modified_image)
    cv2.imshow('mask', modified_mask)
    cv2.waitKey(10)

