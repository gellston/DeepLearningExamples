import os
import cv2 as cv2
import face_recognition
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler

# crawer setting
folder_name4collect = 'D://portrait-dataset//selfie_collect//'
folder_name4save = 'D://portrait-dataset//selfie_crop//'
start_year = 2011
period = 1
image_width = 256
image_height = 256
max_file_count = 1000
# crawer setting

classCount = 0
month = 1
for year in range(period):
    for month in range(11):

        google_crawler = GoogleImageCrawler(storage={'root_dir': folder_name4collect})
        google_crawler.crawl(keyword='selfie',  filters={'date': ((start_year + year,  month, 1), (start_year + year, month, 28)), 'license':'commercial,modify'}, max_num=max_file_count, file_idx_offset='auto')

        fileList = os.listdir(folder_name4collect)
        for fileName in fileList:
            if fileName == '.DS_Store' : continue
            classCount +=1
            filePath = folder_name4collect + fileName
            filename, file_extension = os.path.splitext(filePath)
            modifiedPath = folder_name4save + str(classCount) + '.jpg'
            if file_extension != '.jpg' and file_extension != '.bmp' and file_extension != '.gif' and file_extension != '.jpeg' and file_extension != '.png':
                os.remove(filePath)
                continue
            original = cv2.imread(filePath)
            if original is None:
                classCount -=1
                os.remove(filePath)
                continue
            if len(original.shape) != 3:
                classCount -=1
                os.remove(filePath)
                continue
            image = face_recognition.load_image_file(filePath)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 2 or len(face_locations) == 0:
                classCount -= 1
                os.remove(filePath)
                continue
            resizeImage = cv2.resize(original, (image_width, image_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(modifiedPath, resizeImage)
            cv2.imshow('resize', resizeImage)
            cv2.waitKey(10)
            os.remove(filePath)