
import cv2
import numpy as np
import torch
import torch.nn as nn

# from notUsed.lcn import LecunLCN

LCN_enable = False

def input_filled_mirroring(x, e = 62):      # fill missing data by mirroring the input image
    '''input size 636 --> output size 512'''
    # w, h = x.shape
    w, h = np.shape(x)[0], np.shape(x)[1]
    #e = 62  # extra width on 1 edge
    y = np.zeros((h + e * 2, w + e * 2))
    y[e:h + e, e:w + e] = x
    y[e:e + h, 0:e] = np.flip(y[e:e + h, e:2 * e], 1)  # flip vertically
    y[e:e + h, e + w:2 * e + w] = np.flip(y[e:e + h, w:e + w], 1)  # flip vertically
    y[0:e, 0:2 * e + w] = np.flip(y[e:2 * e, 0:2 * e + w], 0)  # flip horizontally
    y[e + h:2 * e + h, 0:2 * e + w] = np.flip(y[h:e + h, 0:2 * e + w], 0)  # flip horizontally
    return y

def nearest_cells_distance(pos, img):
    # searching in 8 directions to find 2 nearest distances to neighbor cells
    curr_value = img[pos]
    w = img.shape[0];h = img.shape[1]
    count_detect = 0  # detect 2 nearest neighbors. break the search if count_detect = 2
    x, y = pos

    # print("w,h: ", (w, h))
    d1 = 9999999;d2 = 9999999;d3 = 9999999;d4 = 9999999
    d5 = 9999999;d6 = 9999999;d7 = 9999999;d8 = 9999999
    for i in range(1, max(w, h)):
        # (x+1, y), (x+1, y+1), (x+1, y-1),
        # (x, y+1), (x, y-1)
        # (x-1, y), (x-1,y+1), (x-1,y-1),
        if x + i < w:
            if img[(x + i, y)] != curr_value and d1 == 9999999:
                d1 = i; count_detect += 1
            if y + i < h:
                if img[(x + i, y + i)] != curr_value and d2 == 9999999:
                    d2 = i; count_detect += 1
            elif y - i >= 0:
                if img[(x + i, y - i)] != curr_value and d3 == 9999999:
                    d3 = i; count_detect += 1
        if y + i < h:
            if img[(x, y + i)] != curr_value and d4 == 9999999:
                d4 = i; count_detect += 1
        if y - i >= 0:
            if img[(x, y - i)] != curr_value and d5 == 9999999:
                d5 = i; count_detect += 1
        if x - i >= 0:
            if img[(x - i, y)] != curr_value and d6 == 9999999:
                d6 = i; count_detect += 1
            if y + i < h:
                if img[(x - i, y + i)] != curr_value and d7 == 9999999:
                    d7 = i; count_detect += 1
            elif y - i >= 0:
                if img[(x - i, y - i)] != curr_value and d8 == 9999999:
                    d8 = i; count_detect += 1
        if count_detect >= 2: break

    d = [d1, d2, d3, d4, d5, d6, d7, d8]
    min_index = np.argmin(d)
    dist1 = d[min_index]
    d.remove(dist1)
    min_index = np.argmin(d)
    dist2 = d[min_index]
    return dist1, dist2

def create_weight_map(imgs_mask):
        print('-' * 30)
        print('Creating weights for masks ...')
        print('-' * 30)
        weight_map = np.ndarray(
            (len(imgs_mask), 1, np.shape(imgs_mask)[2], np.shape(imgs_mask)[3]), dtype=np.float32)
        w_0 = 10; sigma = 4
        imgs_w, imgs_h = np.shape(imgs_mask)[2], np.shape(imgs_mask)[3]
        total_pixels = imgs_w*imgs_h
        import time
        start = time.time()
        for i in range(len(imgs_mask)):
            if i%10 == 0: print("Done processing: %d images, time passed: %.2f mins" % (i, (time.time() - start)/60.0))
            temp = imgs_mask[i,0,:,:]
            w_border = 1 - np.sum(temp)/total_pixels        # w_c of pixel on the border (black)
            for m in range(imgs_w):
                for n in range(imgs_h):
                    # loop through all pixel of the mask
                    d1,d2 = nearest_cells_distance((m,n), temp)
                    w_c = w_border
                    if temp[m,n] == 0: w_c = 1 - w_border
                    weight_map[i,0,m,n] = w_c*5 + w_0*np.exp(-(d1 + d2)**2/(2*sigma**2))
                    # print("w_0*np.exp(): ", w_0*np.exp(-(d1 + d2)**2/(2*sigma**2)))
                    # print("w_border: ", w_border)

        np.save('data/dataset/weight_map_train.npy', weight_map)

        return weight_map

def load_train_data(npy_path = "data/dataset", hist_equal = False):

    print('-' * 30)
    print('load train images...')

    imgs_train = np.load(npy_path + "/imgs_train.npy")
    imgs_mask_train = np.load(npy_path + "/imgs_mask_train.npy")

    if hist_equal: imgs_train = hist_equalization(imgs_train)

    # weight_map_train = np.load(npy_path + "/weight_map_train.npy")
    # weight_map_train = weight_map_train.astype('float32')

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    if LCN_enable:
        imgs_train = imgs_train.astype('float64')
        imgs_train = LecunLCN(imgs_train, image_shape=imgs_train.shape).output.eval()
        imgs_train = imgs_train.astype('float32')

    if not LCN_enable: imgs_train /= 255

    imgs_train -= imgs_train.mean(axis=0)
    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0

    # weight_map_train = create_weight_map(imgs_mask_train)

    print("Done loading traing data")
    print('-' * 30)
    return imgs_train, imgs_mask_train

def load_val_data(npy_path = "data/dataset", hist_equal = True):

    print('-' * 30)
    print('load validation images...')

    imgs_train = np.load(npy_path + "/imgs_val.npy")
    imgs_mask_train = np.load(npy_path + "/imgs_mask_val.npy")

    if hist_equal: imgs_train = hist_equalization(imgs_train)

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    if LCN_enable:
        imgs_train = imgs_train.astype('float64')
        imgs_train = LecunLCN(imgs_train, image_shape=imgs_train.shape).output.eval()
        imgs_train = imgs_train.astype('float32')

    if not LCN_enable: imgs_train /= 255
    imgs_train -= imgs_train.mean(axis=0)
    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    print("Done loading validation data")
    print('-' * 30)
    return imgs_train, imgs_mask_train

def load_test_data(npy_path = "data/dataset", hist_equal = True):

    print('-' * 30)
    print('load test images...')

    imgs_test = np.load(npy_path + "/imgs_test.npy")

    if hist_equal: imgs_test = hist_equalization(imgs_test)

    imgs_test = imgs_test.astype('float32')

    if LCN_enable:
        imgs_test = imgs_test.astype('float64')
        imgs_test = LecunLCN(imgs_test, image_shape=imgs_test.shape).output.eval()
        imgs_test = imgs_test.astype('float32')

    if not LCN_enable: imgs_test /= 255
    imgs_test -= imgs_test.mean(axis=0)

    print("Done loading test data")
    print('-' * 30)
    return imgs_test


def load_val_5imgs(npy_path = "data/dataset"):
    print('-' * 30)
    print('load train images...')

    imgs_train = np.load(npy_path + "/imgs_val_5imgs.npy")
    imgs_mask_train = np.load(npy_path + "/imgs_mask_val_5imgs.npy")

    # imgs_train = hist_equalization(imgs_train)

    imgs_train = imgs_train.astype('float64')
    imgs_mask_train = imgs_mask_train.astype('float64')

    if LCN_enable: imgs_train = LecunLCN(imgs_train, image_shape=imgs_train.shape).output.eval()

    if not LCN_enable: imgs_train /= 255

    mean = imgs_train.mean(axis=0)
    imgs_train -= mean


    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0

    # weight_map_train = self.create_weight_map(imgs_mask_train)

    print("Done loading traing data")
    print('-' * 30)
    return imgs_train, imgs_mask_train

def hist_equalization(inputs):      # inputs are uint8, 0-255, Nx1x636x636
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    output = inputs.copy()
    for i in range(len(inputs)):
        img = inputs[i, 0, :, :]
        # res = cv2.equalizeHist(img)
        res = clahe.apply(img)
        output[i,0,:,:] = res
    return output

### BinaryCrossEntropyLoss2d().forward(logits, labels)
class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """
        Binary cross entropy loss 2D
        Args:
            weight:
            size_average:
        """
        super(BinaryCrossEntropyLoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        # probs = F.sigmoid(logits)
        probs = logits
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)      # (output, labels)

class BCE_with_Weights(nn.Module):
    def __init__(self, weights=None):
        """
        Binary cross entropy loss 2D
        Args:
            weights:
        """
        super(BCE_with_Weights, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        if self.weights is not None:
            loss = self.weights * (target * torch.log(output) + (1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))

