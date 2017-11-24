import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Unet
import Unet_model
import tifffile as tiff
import Utils
import time
import matplotlib.pyplot as plt
import split_merge_tif
import cv2
import Analysis

# npy_path = "/Users/mac/Documents/cse527 data/DataAugmentation/data/dataset/40aug_10ela_3inten_reflect/"
# npy_path = "/home/anhxtuan/HanLe/dataset/40aug_10ela_3inten_reflect/"
# # names = ['imgs_train', 'imgs_mask_train', 'imgs_val', 'imgs_mask_val']
# names = ['imgs_train', 'imgs_mask_train']
# for i in range(len(names)):
#     imgs_train = np.load(npy_path + names[i] + '.npy')
#     print(imgs_train.shape)
#     for j in range(5):
#         imgs = np.ndarray((np.int32(imgs_train.shape[0]/5), 1, imgs_train.shape[2], imgs_train.shape[3]), dtype=np.uint8)
#         print(imgs.shape)
#         names_j = names[i] + "_" + str(j+1) + '.npy'
#         imgs[:,0,:,:] = imgs_train[j*imgs.shape[0]:(j+1)*imgs.shape[0],0,:,:]
#         np.save(npy_path + names_j, imgs)
#
#
#
# x = Variable(torch.randn(1,1,700, 700))       # simulate input image

path = "/home/anhxtuan/HanLe/dataset/20aug_10ela_3inten_reflect"
imgs_val, imgs_mask_val = Utils.load_val_data(npy_path=path)

valset = zip(imgs_val, imgs_mask_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                        shuffle=False, num_workers=2)
# net = Unet.UNet()                     # Unet 1

Analysis.analysis("Leg21_model_saved", valloader)





# x = inputs.numpy()[:, :, ci[0]:ci[0] + 572, ci[1]:ci[1] + 572]
# y = labels.numpy()[:, :, cl[0]: cl[0] + 388, cl[1]:cl[1] + 388]

#
# out = net(x)


# lena_lcn = LecunLCN(img, image_shape = img.shape).output.eval()
# x = lena_lcn[0,0,:,:]
# plt.imshow(x, cmap = 'gray')
# plt.show()


# imgs_lcn = LecunLCN(imgs_train, image_shape = imgs_train.shape).output.eval()

# mean = imgs_lcn.mean(axis=0)
# out = imgs_lcn - mean
#
# inputs_lcn = imgs_train[:, :, 62:574, 62:574]
# N = 5
# for t in range(0):
#     # t = np.random.randint(N)
#     img_train = imgs_train[t, 0, 62:574, 62:574]
#     img_mask = imgs_mask_train[t, 0, :, :]
#
#     blur = cv2.GaussianBlur(img_train, (5, 5), 0)
#     smooth = cv2.addWeighted(blur, 1.5, img_train, -0.5, 0)
#     inputs_lcn[t, 0, :, :] = smooth
#
#     plt.subplot(131);
#     plt.imshow(img_train, cmap='gray');
#     plt.title("image " + str(t + 1))
#     plt.subplot(132)
#     plt.imshow(smooth, cmap='gray')
#     plt.subplot(133);
#     plt.imshow(LecunLCN(inputs_lcn, image_shape=inputs_lcn.shape).output.eval()[t, 0, :, :], cmap='gray')
#
#     #     plt.figure(figsize=(8,10))
#     #     plt.subplot(131); plt.imshow(img_train, cmap='gray'); plt.title("image " + str(t+1))
#     #     plt.subplot(132); plt.imshow(imgs_lcn[t,0,62:574,62:574], cmap='gray'); plt.title("LCN WITH hist equalization")
#     #     plt.subplot(133); plt.imshow(img_mask, cmap='gray')
#
#     plt.show()
