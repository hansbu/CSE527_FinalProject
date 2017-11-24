from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import Unet_model
import tifffile as tiff
import Utils
import time
import matplotlib.pyplot as plt
import numpy as np
import sys

def help_message():
   print("Usage: [File_name] [Input_image_path] [label_image]")
   print("Ignore label_image if input_image is a test image without label")
   print(sys.argv[0] + "data/train/0.tif" + "data/label/0.tif")


def to_var(x):
    if torch.cuda.is_available():
        return Variable(x.cuda())
    else:
        return Variable(x)

def NetPredict_image(net, input_image):
    net.eval()
    # input_image is grayscale input image of 512x512
    data = Utils.input_filled_mirroring(input_image, e = 62)
    imgdatas = np.ndarray((1, 1, data.shape[0], data.shape[1]), dtype=np.uint8)
    imgdatas[0] = np.expand_dims(data, 0)
    data = Utils.hist_equalization(imgdatas)
    data = data.astype('float32')
    data /= 255
    data = torch.from_numpy(data)
    data = to_var(data)
    out = net(data)
    out = F.sigmoid(out)
    img = out.cpu()
    img = img[0, 0, :, :]
    img = img.data.numpy()
    img_shape = img.shape
    img = img.reshape(-1)
    img[img < 0.5] = 0
    img[img >= 0.5] = 1
    img = img.reshape(img_shape)
    img = np.uint8(img * 255)
    return img


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = Unet_model.UNet_BN()
        net.load_state_dict(torch.load("Leg21_model_saved"))
        net.cuda()
    else:
        net = Unet_model.UNet_BN()
        net.load_state_dict(torch.load("Leg21_model_saved", map_location=lambda storage, loc: storage))

    # Validate the input arguments
    if (len(sys.argv) == 3):
        input_imgage = sys.argv[1]
        label_image = sys.argv[2]
        img = tiff.imread(input_imgage)
        lable = tiff.imread(label_image)
        start = time.time()
        pred = NetPredict_image(net, img)
        print("Processing time: %.2f", time.time() - start)
        plt.figure(figsize=(12, 8))
        plt.subplot(131); plt.imshow(img, cmap='gray'); plt.title("input image")
        plt.subplot(132); plt.imshow(pred, cmap='gray'); plt.title("prediction")
        plt.subplot(133); plt.imshow(lable, cmap='gray'); plt.title("label")
        plt.show()
    if (len(sys.argv) == 2):
        input_imgage = sys.argv[1]
        img = tiff.imread(input_imgage)
        start = time.time()
        pred = NetPredict_image(net, img)
        print("Processing time: %.2f", time.time() - start)
        plt.figure(figsize=(12, 8))
        plt.subplot(121); plt.imshow(img, cmap='gray'); plt.title("input image")
        plt.subplot(122); plt.imshow(pred, cmap='gray'); plt.title("prediction")
        plt.show()
    else:
        help_message()
        sys.exit()

