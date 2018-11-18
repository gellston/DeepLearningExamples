import os
import cv2 as cv2
import numpy as np

from models.model_custom_densenet_segmentation_v1 import model_custom_densenet_segmentation_v1
from util.segmentation_dataloader_v1 import segmentation_dataloader_v1

sample_loader = segmentation_dataloader_v1('D://portrait-dataset//train_input256x256//', 'D://portrait-dataset//train_label256x256//')
augmentation_image_path = 'D://portrait-dataset//train_input256x256(augmentation)//'
augmentation_label_path = 'D://portrait-dataset//train_label256x256(augmentation)//'
background_sample_path = 'D://portrait-dataset//backgrounds//'

stretch_factor = 1.3
sample_size = sample_loader.size()

loop_count = 0
for loop in range(sample_size):
    loop_count += 1
    filename = str(loop_count)

    input_images, input_labels = sample_loader.load([256, 256, 3], [256, 256, 1], 1, 1, 1)
    cv2.imshow('input_image(original)', input_images[0])
    cv2.imshow('input_label(original)', input_labels[0])
    cv2.moveWindow("input_image(original)", 0, 0);
    cv2.moveWindow("input_label(original)", 0, 256);

    height, width, _ = input_images[0].shape

    flip_without_background_input = cv2.flip(input_images[0], 1)
    flip_without_background_output = cv2.flip(input_labels[0], 1)
    cv2.imshow('flip_image(original)', flip_without_background_input)
    cv2.imshow('flip_label(original)', flip_without_background_output)
    cv2.moveWindow("flip_image(original)", 256, 0);
    cv2.moveWindow("flip_label(original)", 256, 256);

    input_stretched_width = cv2.resize(input_images[0], (int(width * stretch_factor), height))[0:256, 0:256]
    label_stretched_width = cv2.resize(input_labels[0], (int(width * stretch_factor), height))[0:256, 0:256]
    cv2.imshow('input_stretched(width)', input_stretched_width)
    cv2.imshow('label_stretched(width)', label_stretched_width)
    cv2.moveWindow("input_stretched(width)", 512, 0);
    cv2.moveWindow("label_stretched(width)", 512, 256);

    input_stretched_height = cv2.resize(input_images[0], (width, int(height * stretch_factor)))[0:256, 0:256]
    label_stretched_height = cv2.resize(input_labels[0], (width, int(height * stretch_factor)))[0:256, 0:256]
    cv2.imshow('input_stretched(height)', input_stretched_height)
    cv2.imshow('label_stretched(height)', label_stretched_height)
    cv2.moveWindow("input_stretched(height)", 768, 0);
    cv2.moveWindow("label_stretched(height)", 768, 256);

    input_stretched_width_flip = cv2.resize(flip_without_background_input, (int(width * stretch_factor), height))[0:256, 0:256]
    label_stretched_width_flip = cv2.resize(flip_without_background_output, (int(width * stretch_factor), height))[0:256, 0:256]
    cv2.imshow('input_stretched(width)_flip', input_stretched_width_flip)
    cv2.imshow('label_stretched(width)_flip', label_stretched_width_flip)
    cv2.moveWindow("input_stretched(width)_flip", 1024, 0);
    cv2.moveWindow("label_stretched(width)_flip", 1024, 256);

    input_stretched_height_flip = cv2.resize(flip_without_background_input, (width, int(height * stretch_factor)))[0:256, 0:256]
    label_stretched_height_flip = cv2.resize(flip_without_background_output, (width, int(height * stretch_factor)))[0:256, 0:256]
    cv2.imshow('input_stretched(height)_flip', input_stretched_height_flip)
    cv2.imshow('label_stretched(height)_flip', label_stretched_height_flip)
    cv2.moveWindow("input_stretched(height)_flip", 1280, 0);
    cv2.moveWindow("label_stretched(height)_flip", 1280, 256);

    cv2.imwrite(augmentation_image_path + filename + "_flip.jpg", flip_without_background_input)
    cv2.imwrite(augmentation_label_path + filename + "_flip.jpg", flip_without_background_output)

    cv2.imwrite(augmentation_image_path + filename + "_stretch_width.jpg", input_stretched_width)
    cv2.imwrite(augmentation_label_path + filename + "_stretch_width.jpg", label_stretched_width)

    cv2.imwrite(augmentation_image_path + filename + "_stretch_height.jpg", input_stretched_height)
    cv2.imwrite(augmentation_label_path + filename + "_stretch_height.jpg", label_stretched_height)

    cv2.imwrite(augmentation_image_path + filename + "_stretch_width_flip.jpg", input_stretched_width_flip)
    cv2.imwrite(augmentation_label_path + filename + "_stretch_width_flip.jpg", label_stretched_width_flip)

    cv2.imwrite(augmentation_image_path + filename + "_stretch_height_flip.jpg", input_stretched_height_flip)
    cv2.imwrite(augmentation_label_path + filename + "_stretch_height_flip.jpg", label_stretched_height_flip)



    #masked_flip_without_background_input = cv2.bitwise_and(input_stretched_width, mask3channel)
    #cv2.imshow('input_stretched(height)_flip + background', masked_flip_without_background_input)
    #cv2.moveWindow("input_stretched(height)_flip + background", 1530, 0);


    """"
    background_count = 0
    for background_filename in os.listdir(background_sample_path):
        if background_filename == '.DS_Store': continue
        background_count += 1
        loaded_background = cv2.imread(background_sample_path + background_filename)
        resized_background = cv2.resize(loaded_background, (256, 256))
        resized_background = cv2.GaussianBlur(resized_background, (3, 3), 0)


        mask3channel = np.zeros_like(input_images[0])
        mask3channel[:, :, 0] = flip_without_background_output
        mask3channel[:, :, 1] = flip_without_background_output
        mask3channel[:, :, 2] = flip_without_background_output

        mask3channel_inv = cv2.bitwise_not(mask3channel)

        foreground = cv2.bitwise_and(flip_without_background_input, mask3channel)
        background = cv2.bitwise_and(resized_background, mask3channel_inv)
        attached_image = cv2.add(foreground, background)
        cv2.imwrite(augmentation_image_path + filename + "_flip_background_" + str(background_count) + ".jpg", attached_image)
        cv2.imwrite(augmentation_label_path + filename + "_flip_background_" + str(background_count) + ".jpg", flip_without_background_output)


        mask3channel = np.zeros_like(input_images[0])
        mask3channel[:, :, 0] = label_stretched_width
        mask3channel[:, :, 1] = label_stretched_width
        mask3channel[:, :, 2] = label_stretched_width

        mask3channel_inv = cv2.bitwise_not(mask3channel)

        foreground = cv2.bitwise_and(input_stretched_width, mask3channel)
        background = cv2.bitwise_and(resized_background, mask3channel_inv)
        attached_image = cv2.add(foreground, background)
        cv2.imwrite(augmentation_image_path + filename + "_stretched_width_background_" + str(background_count) + ".jpg", attached_image)
        cv2.imwrite(augmentation_label_path + filename + "_stretched_width_background_" + str(background_count) + ".jpg", label_stretched_width)



        mask3channel = np.zeros_like(input_images[0])
        mask3channel[:, :, 0] = label_stretched_height
        mask3channel[:, :, 1] = label_stretched_height
        mask3channel[:, :, 2] = label_stretched_height

        mask3channel_inv = cv2.bitwise_not(mask3channel)

        foreground = cv2.bitwise_and(input_stretched_height, mask3channel)
        background = cv2.bitwise_and(resized_background, mask3channel_inv)
        attached_image = cv2.add(foreground, background)
        cv2.imwrite(augmentation_image_path + filename + "_stretched_height_background_" + str(background_count) + ".jpg", attached_image)
        cv2.imwrite(augmentation_label_path + filename + "_stretched_height_background_" + str(background_count) + ".jpg", label_stretched_height)



        mask3channel = np.zeros_like(input_images[0])
        mask3channel[:, :, 0] = label_stretched_width_flip
        mask3channel[:, :, 1] = label_stretched_width_flip
        mask3channel[:, :, 2] = label_stretched_width_flip

        mask3channel_inv = cv2.bitwise_not(mask3channel)

        foreground = cv2.bitwise_and(input_stretched_width_flip, mask3channel)
        background = cv2.bitwise_and(resized_background, mask3channel_inv)
        attached_image = cv2.add(foreground, background)
        cv2.imwrite(augmentation_image_path + filename + "_stretched_width_flip_background_" + str(background_count) + ".jpg", attached_image)
        cv2.imwrite(augmentation_label_path + filename + "_stretched_width_flip_background_" + str(background_count) + ".jpg", label_stretched_width_flip)



        mask3channel = np.zeros_like(input_images[0])
        mask3channel[:, :, 0] = label_stretched_height_flip
        mask3channel[:, :, 1] = label_stretched_height_flip
        mask3channel[:, :, 2] = label_stretched_height_flip

        mask3channel_inv = cv2.bitwise_not(mask3channel)

        foreground = cv2.bitwise_and(input_stretched_height_flip, mask3channel)
        background = cv2.bitwise_and(resized_background, mask3channel_inv)
        attached_image = cv2.add(foreground, background)
        cv2.imwrite(augmentation_image_path + filename + "_stretched_height_flip_background_" + str(background_count) + ".jpg", attached_image)
        cv2.imwrite(augmentation_label_path + filename + "_stretched_height_flip_background_" + str(background_count) + ".jpg", label_stretched_height_flip)
    """
    cv2.waitKey(10)