from typing import Any, Union

import cv2 as cv2
import os

from icrawler.builtin import GoogleImageCrawler

# google_crawler = GoogleImageCrawler(storage={'root_dir': 'animal/dog/'})
# google_crawler.crawl('dog', max_num=15000, max_size=(1000,1000))

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += "/animal_train"

symbol = len(os.listdir(dir_path))

classCount = 0
for className in os.listdir(dir_path):
    classCount = 0
    fullPath = dir_path + "/" + className;

    google_crawler = GoogleImageCrawler(storage={'root_dir': 'animal_train/' + className + '/'})
    google_crawler.crawl(className, max_num=15000, min_size=(100, 100))

    classCount = classCount + 1
    fileIndex = 0

    if os.path.isdir(fullPath):
        fileList = os.listdir(fullPath)
        for fileName in fileList:
            if fileName == '.DS_Store' : continue

            filePath = fullPath + "/" + fileName
            filename, file_extension = os.path.splitext(filePath)
            if file_extension != '.jpg':
                os.remove(filePath)
                continue

            fileIndex = fileIndex + 1

            modifiedPath = fullPath + "/" + str(fileIndex) + "_" + str(classCount) + "_" + className + file_extension
            print(modifiedPath)
            original = cv2.imread(filePath)

            if original is None:
                os.remove(filePath)
                fileIndex = fileIndex = - 1
                continue

            resizeImage = cv2.resize(original, (224,224), interpolation=cv2.INTER_AREA)

            cv2.imwrite(modifiedPath, resizeImage)
            cv2.imshow('resize', resizeImage)
            cv2.waitKey(10)
            os.remove(filePath)
