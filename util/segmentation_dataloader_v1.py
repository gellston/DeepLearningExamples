import os
import cv2 as cv2
import numpy as np


class segmentation_dataloader_v1:
    def __init__(self, image_path, label_path):
        self.input_images_path = image_path
        self.label_path = label_path

        self.input_images_paths = []
        self.label_paths = []
        self.dataset_count = 0
        self.currentIndex = 0

        filelist = sorted(os.listdir(self.input_images_path))
        for filename in filelist:
            if filename == '.DS_Store': continue
            temp1 = self.input_images_path + '/' + filename
            temp2 = self.label_path + '/' + filename
            self.input_images_paths.append(temp1)
            self.label_paths.append(temp2)
            self.dataset_count += 1



    def load(self, shape1, shape2, devide1, devide2, batch):
        images = []
        labels = []

        for index in range(batch):
            if index + self.currentIndex >= self.dataset_count:
                return (None, None)

            image_path = self.input_images_paths[self.currentIndex + index]
            image = cv2.imread(image_path).astype(np.uint8)
            npImage = np.array(image)
            npImage = npImage / devide1
            npImage = npImage.flatten().reshape(shape1)
            npImage = np.array(npImage, dtype=np.uint8)
            images.append(npImage)


            label_path = self.label_paths[self.currentIndex + index]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            ret, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            npLabel = np.array(label)
            npLabel = npLabel / devide2
            npLabel = npLabel.flatten().reshape(shape2)
            npLabel = np.array(npLabel, dtype=np.uint8)

            labels.append(npLabel)

            if index + self.currentIndex >= self.dataset_count:
                break

        self.currentIndex += batch

        return (images, labels)

    def clear(self):
        self.currentIndex = 0

    def size(self):
        return self.dataset_count

