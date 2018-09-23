import os
import cv2 as cv2


from icrawler.builtin import GoogleImageCrawler


folder_name = 'animal_train'
start_year = 2010
period = 1
image_width = 224
image_height = 224
max_file_count = 1000


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/'
dir_path += folder_name

symbol = len(os.listdir(dir_path))

classCount = 0
for className in os.listdir(dir_path):
    if className == '.DS_Store': continue
    classCount = 0
    fullPath = dir_path + "/" + className;

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

            resizeImage = cv2.resize(original, (image_width,image_height), interpolation=cv2.INTER_AREA)

            cv2.imwrite(modifiedPath, resizeImage)
            cv2.imshow('resize', resizeImage)
            cv2.waitKey(10)
            os.remove(filePath)
