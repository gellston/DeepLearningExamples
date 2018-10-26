import os
import cv2 as cv2
import numpy as np
import tensorflow as tf

from icrawler.builtin import GoogleImageCrawler

# crawer setting
folder_name = 'dataset/animal-validation-v2'
start_year = 2011
period = 1
image_width = 224
image_height = 224
max_file_count = 1000
on_google_crawler = False
on_filter_by_tensor = False
on_filter_target_percent = 0.70
label = {'cat': 0, 'dog': 1, 'elephant': 2, 'giraffe':3, 'horse':4}

# init tensor network
sess = tf.Session()
saver = tf.train.import_meta_graph('./pretrained-models/animal_trained-model(v1)/animal_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./pretrained-models/animal_trained-model(v1)'))
graph = tf.get_default_graph()
output = graph.get_tensor_by_name("AnimalClassifier/output:0")
input = graph.get_tensor_by_name("AnimalClassifier/input:0")
dropout = graph.get_tensor_by_name("AnimalClassifier/dropout:0")

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/'
dir_path += folder_name
symbol = len(os.listdir(dir_path))
classCount = 0

for className in os.listdir(dir_path):

    if className == '.DS_Store': continue

    classCount = 0
    fullPath = dir_path + "/" + className;

    if on_google_crawler:
        google_crawler = GoogleImageCrawler(storage={'root_dir': folder_name + '/' + className + '/'})

        for year in range(period):
           for month in range(11):
               month += 1
               google_crawler.crawl(keyword=className,
                                    filters={'date': ((start_year + year,  month, 1), (start_year + year, month, 28))},
                                    max_num=max_file_count,
                                    file_idx_offset='auto')
    classCount = classCount + 1
    fileIndex = 0

    if os.path.isdir(fullPath):
        fileList = os.listdir(fullPath)
        for fileName in fileList:
            if fileName == '.DS_Store' : continue

            filePath = fullPath + "/" + fileName
            filename, file_extension = os.path.splitext(filePath)
            if file_extension != '.jpg' and file_extension != '.bmp' and file_extension != '.gif' and file_extension != '.jpeg' and file_extension != '.png':
                os.remove(filePath)
                continue

            fileIndex = fileIndex + 1
            modifiedPath = fullPath + "/" + str(fileIndex) + "_" + str(classCount) + "_" + className + '.jpg'
            original = cv2.imread(filePath)

            if original is None:
                os.remove(filePath)
                fileIndex = fileIndex = fileIndex - 1
                continue

            if len(original.shape) != 3:
                os.remove(filePath)
                fileIndex = fileIndex = fileIndex - 1
                continue

            resizeImage = cv2.resize(original, (image_width, image_height), interpolation=cv2.INTER_AREA)
            npImage = []

            if on_filter_by_tensor:
                temp = np.array(resizeImage)
                temp = temp = temp.flatten().reshape([image_width * image_height * 3])
                npImage.append(temp)
                prediction = sess.run(output, {input: npImage, dropout: 1})
                print('label:', className, ':', prediction[0][label[className]], '\n')
                if prediction[0][label[className]] < on_filter_target_percent:
                    os.remove(filePath)
                    continue


            cv2.imwrite(modifiedPath, resizeImage)
            cv2.imshow('resize', resizeImage)
            cv2.waitKey(10)
            os.remove(filePath)
