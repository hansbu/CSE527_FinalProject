from __future__ import print_function, division
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from Utils import *
import split_merge_tif as split_merge
import torch
import elastic_transform as ET

No_img = 50         # number of augmentated image per set = No_img + 1
class myAugmentation(object):
    """
	A class used to augmentate image
	Firstly, read train image and label seperately, and then merge them together for the next process

	Secondly, use keras preprocessing to augmentate image

	Finally, seperate augmentated image apart into train image and label
	"""

    def __init__(self,
                 train_path="data/train",
                 label_path="data/label",
                 merge_path="data/merge",
                 aug_merge_path="data/aug_merge",
                 aug_train_path="data/aug_train",
                 aug_label_path="data/aug_label",
                 img_type="tif"):
        """
		Using glob to get all .img_type form path
		"""

        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        # self.datagen = ImageDataGenerator(
        #     rotation_range=0.2,
        #     width_shift_range=0.05,
        #     height_shift_range=0.05,
        #     shear_range=0.05,
        #     zoom_range=0.05,
        #     horizontal_flip=True,
        #     vertical_flip=True,
        #     fill_mode='nearest')
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.08,
            zoom_range=0.08,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

    def Augmentation(self):
        """
		Start augmentation.....
		"""
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0

        print("len of trains: ", len(trains))
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]     # last channel of x_t is label --> x_t is called merged img

            # plt.subplot(121); plt.imshow(x_t[:, :, 0], cmap='gray')
            # plt.subplot(122); plt.imshow(x_t[:, :, 2], cmap='gray')
            # plt.show()

            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)
            img = x_t
            img = img.reshape((1,) + img.shape)

            savedir = path_aug_merge + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i))

    def doAugmentate(self,
                     img,
                     save_to_dir,
                     save_prefix,
                     batch_size=1,
                     save_format='tif',
                     imgnum=No_img):

        # augmentate one image

        datagen = self.datagen
        i = 0
        for batch in datagen.flow(
                img,
                batch_size=batch_size,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):
        """
		split merged image apart
		"""
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path
        for i in range(self.slices):        # 30 images
            path = path_merge + "/" + str(i)
            train_imgs = glob.glob(path + "/*." + self.img_type)    # add subfolder 0 --> 29
            savedir = path_train + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            savedir = path_label + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex(
                    "." + self.img_type)]
                img = cv2.imread(imgname)
                img_train = img[:, :, 2]  #cv2 read image rgb->bgr
                img_label = img[:, :, 0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train"
                            + "." + self.img_type, img_train)
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label"
                            + "." + self.img_type, img_label)

    def splitTransform(self):
        """
		split perspective transform images
		"""
        path_merge = "data/deform/deform_norm2"
        path_train = "data/deform/train/"
        path_label = "data/deform/label/"
        train_imgs = glob.glob(path_merge + "/*." + self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/") + 1:imgname.rindex(
                "." + self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:, :, 2]  #cv2 read image rgb->bgr
            img_label = img[:, :, 0]
            cv2.imwrite(path_train + midname + "." + self.img_type, img_train)
            cv2.imwrite(path_label + midname + "." + self.img_type, img_label)


class dataProcess(object):

    def __init__(self,
                 out_rows,
                 out_cols,
                 data_path="data/aug_train",
                 label_path="data/aug_label",
                 test_path="data/test",
                 npy_path="data/dataset",
                 img_type="tif",
                 img_No_train=0,
                 img_No_val = 0,
                 extra_padding = 124):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

        self.img_No_train = (No_img+1)*25
        self.img_No_val = (No_img + 1) * 30 - self.img_No_train
        self.extra_padding = extra_padding

    def create_val_5imgs(self):
        extra = self.extra_padding
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        imgdatas_val = np.ndarray(
            (5, 1, self.out_rows + extra, self.out_cols + extra), dtype=np.uint8)
        imglabels_val = np.ndarray(
            (5, 1, self.out_rows, self.out_cols), dtype=np.uint8)

        index = 0
        for i in range(25,30):
            train_img_path = "data/train/" + str(i) + ".tif"
            label_img_path = "data/label/" + str(i) + ".tif"
            img = load_img(train_img_path, grayscale=True)
            label = load_img(label_img_path, grayscale=True)

            img = np.array(img)
            label = np.array(label)

            img = input_filled_mirroring(img)
            img = np.expand_dims(img,0)
            label = np.expand_dims(label,0)
            imglabels_val[index] = label # save validation data
            imgdatas_val[index] = img
            index += 1
        np.save(self.npy_path + '/imgs_val_5imgs.npy', imgdatas_val)
        np.save(self.npy_path + '/imgs_mask_val_5imgs.npy', imglabels_val)
        print('Saving to .npy files done.')

    def load_val_5imgs(self):
        print('-' * 30)
        print('load train images...')

        imgs_train = np.load(self.npy_path + "/imgs_val_5imgs.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_val_5imgs.npy")

        # print 'imgs values: ', imgs_train[0,:,:,0]

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0

        # weight_map_train = self.create_weight_map(imgs_mask_train)

        print("Done loading traing data")
        print('-' * 30)
        return imgs_train, imgs_mask_train

    def create_test_data(self):
        i = 0
        extra = self.extra_padding
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        # imgs = glob.glob(self.test_path + "/*." + self.img_type)
        imgs_len = 30
        imgdatas = np.ndarray(
            (imgs_len, 1, self.out_rows + extra, self.out_cols + extra), dtype=np.uint8)
        for i in range(imgs_len):
            imgname = self.test_path + "/" + str(i) + ".tif"
            img = load_img(imgname, grayscale=True)
            img = np.array(img)
            img = input_filled_mirroring(img)
            # img = img[:, :, np.newaxis]
            imgdatas[i] = np.expand_dims(img,0)
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def create_train_data(self):
        extra = self.extra_padding
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        ET_params = np.array([[2, 0.08, 0.08], [2, 0.05, 0.05], [3, 0.07, 0.09], [3, 0.12, 0.07]]) * self.out_cols
        len_scaled = len(ET_params) + 1

        imgdatas = np.ndarray(
            (self.img_No_train*len_scaled, 1, self.out_rows+extra, self.out_cols+extra), dtype=np.uint8)
        imglabels = np.ndarray(
            (self.img_No_train*len_scaled, 1, self.out_rows, self.out_cols), dtype=np.uint8)

        imgdatas_val = np.ndarray(
            (self.img_No_val*len_scaled, 1, self.out_rows + extra, self.out_cols + extra), dtype=np.uint8)
        imglabels_val = np.ndarray(
            (self.img_No_val*len_scaled, 1, self.out_rows, self.out_cols), dtype=np.uint8)

        index = 0
        import time
        start = time.time()

        for i in range(30):
            train_foldername = self.data_path + "/" + str(i)
            label_foldername = self.label_path + "/" + str(i)
            imgs = glob.glob(train_foldername + "/*." + self.img_type)

            for imgname in imgs:
                # print "imgname: ", imgname
                midname = imgname[imgname.rindex("/") + 1:]
                img_name_only = midname[0:midname.rindex("_")]

                train_img_path = train_foldername + "/" + img_name_only + "_train." + self.img_type
                label_img_path = label_foldername + "/" + img_name_only + "_label." + self.img_type
                img = load_img(train_img_path, grayscale=True)
                label = load_img(label_img_path, grayscale=True)

                img = np.array(img)         # size of 512x512
                label = np.array(label)     # size of 512x512

                #  add elastic transform here!!!
                im_merge = np.concatenate((img[..., None], label[..., None]), axis=2)
                # print("size of im_merge: ", im_merge.shape)
                for k in range(len(ET_params) + 1):
                    if k > 0:   # index 0 is for the original image
                        im_merge_t = ET.elastic_transform(im_merge, ET_params[k-1,0], ET_params[k-1,1],ET_params[k-1,2])
                        # Split image and mask
                        img = im_merge_t[..., 0]
                        label = im_merge_t[..., 1]

                    # original code for only 1 image augmentation
                    img = input_filled_mirroring(img)
                    img = np.expand_dims(img,0)
                    label = np.expand_dims(label,0)
                    if index < self.img_No_train*len_scaled:
                        imglabels[index] = label
                        imgdatas[index] = img
                    else:
                        imglabels_val[index-self.img_No_train*len_scaled] = label # save validation data
                        imgdatas_val[index-self.img_No_train*len_scaled] = img
                    index += 1
                    # print("index: ", index)
                    if (index + 1) % 10 == 0: print("Processed: %d/%d...Time passed: %.5f mins" % (index + 1,
                            self.img_No_train*len_scaled + self.img_No_train*len_scaled, (time.time() - start)/60.0))

        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        np.save(self.npy_path + '/imgs_val.npy', imgdatas_val)
        np.save(self.npy_path + '/imgs_mask_val.npy', imglabels_val)
        print('Saving to .npy files done.')

    def load_train_data(self):
        print('-' * 30)
        print('load train images...')

        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        # weight_map_train = np.load(self.npy_path + "/weight_map_train.npy")
        # weight_map_train = weight_map_train.astype('float32')

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        # weight_map_train = self.create_weight_map(imgs_mask_train)

        print("Done loading traing data")
        print('-' * 30)
        return imgs_train, imgs_mask_train

    def load_val_data(self):
        print('-' * 30)
        print('load validation images...')

        imgs_train = np.load(self.npy_path + "/imgs_val.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_val.npy")

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        print("Done loading validation data")
        print('-' * 30)
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')

        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean

        print("Done loading test data")
        print('-' * 30)
        return imgs_test

    def create_weight_map(self, imgs_mask):
        print('-' * 30)
        print('Creating weights for masks ...')
        print('-' * 30)
        # _, imgs_mask = self.load_train_data()
        weight_map = np.ndarray(
            (len(imgs_mask), 1, np.shape(imgs_mask)[2], np.shape(imgs_mask)[3]), dtype=np.float16)

        def nearest_cells_distance(pos, img):
            # searching in 8 directions to find 2 nearest distances to neighbor cells
            curr_value = img[pos]
            w = img.shape[0]; h = img.shape[1]
            count_detect = 0    # detect 2 nearest neighbors. break the search if count_detect = 2
            x, y = pos

            print("w,h: ", (w,h ))
            d1 = 9999999; d2 = 9999999; d3 = 9999999; d4 = 9999999
            d5 = 9999999; d6 = 9999999; d7 = 9999999; d8 = 9999999
            for i in range(1, max(w,h)):
                # (x+1, y), (x+1, y+1), (x+1, y-1),
                # (x, y+1), (x, y-1)
                # (x-1, y), (x-1,y+1), (x-1,y-1),
                if x + i < w:
                    if temp[(x + i, y)] != curr_value and d1 == 9999999:
                        d1 = i; count_detect += 1
                    if y + i < h:
                        if temp[(x + i, y + i)] != curr_value and d2 == 9999999:
                            d2 = i; count_detect += 1
                    elif y - i >= 0:
                        if temp[(x + i, y - i)] != curr_value and d3 == 9999999:
                            d3 = i; count_detect += 1
                if y + i < h:
                    if temp[(x, y + i)] != curr_value and d4 == 9999999:
                        d4 = i; count_detect += 1
                if y - i >= 0:
                    if temp[(x, y - i)] != curr_value and d5 == 9999999:
                        d5 = i; count_detect += 1
                if x - i >= 0:
                    if temp[(x - i, y)] != curr_value and d6 == 9999999:
                        d6 = i; count_detect += 1
                    if y + i < h:
                        if temp[(x - i, y + i)] != curr_value and d7 == 9999999:
                            d7 = i; count_detect += 1
                    elif y - i >= 0:
                        if temp[(x - i, y - i)] != curr_value and d8 == 9999999:
                            d8 = i; count_detect += 1
                if count_detect >= 2: break

            d = [d1,d2,d3,d4,d5,d6,d7,d8]
            min_index = np.argmin(d)
            dist1 = d[min_index]
            d.remove(dist1)
            min_index = np.argmin(d)
            dist2 = d[min_index]
            return dist1, dist2

        w_0 = 10; sigma = 5
        imgs_w, imgs_h = np.shape(imgs_mask)[2], np.shape(imgs_mask)[3]
        total_pixels = imgs_w*imgs_h
        for i in range(len(imgs_mask)):
            temp = imgs_mask[i,0,:,:]
            w_border = 1 - np.sum(temp)/total_pixels        # w_c of pixel on the border (black)
            for m in range(imgs_w):
                for n in range(imgs_h):
                    # loop through all pixel of the mask
                    d1,d2 = nearest_cells_distance((m,n), temp)
                    w_c = w_border
                    if temp[m,n] == 0: w_c = 1 - w_border
                    weight_map[i,0,m,n] = w_c + w_0*np.exp(-(d1 + d2)**2/(2*sigma**2))
                    # print("w_0*np.exp(): ", w_0*np.exp(-(d1 + d2)**2/(2*sigma**2)))
                    # print("w_border: ", w_border)

        return weight_map

if __name__ == "__main__":
    # split_merge.split_img()     # split raw data
    #
    aug = myAugmentation()
    aug.Augmentation()
    aug.splitMerge()

    mydata = dataProcess(512, 512, extra_padding=124)
    mydata.create_train_data()

    # mydata.create_test_data()
    # mydata.create_weight_map()
    # mydata.create_val_5imgs()



    # imgs_test = mydata.load_test_data()
    # imgs_train,imgs_mask_train = mydata.load_train_data()
    # imgs_val, imgs_mask_val = mydata.load_val_data()
    #
    # print("test image loading: ", imgs_test.shape)
    # print("train image loading: ", imgs_train.shape)
    # print("train image label loading: ", imgs_mask_train.shape)
    # print("val image loading: ", imgs_val.shape)
    # print("val image label loading: ", imgs_mask_val.shape)


    # for i in range(len(imgs_train)):
    #     plt.figure(i)
    #     plt.subplot(121)
    #     plt.imshow(imgs_train[i,0, :, :], cmap='gray')
    #     plt.subplot(122)
    #     plt.imshow(imgs_mask_train[i,0, :, :], cmap='gray')
    #     plt.show()
