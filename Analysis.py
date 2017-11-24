from __future__ import print_function, division
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import Unet_model
import tifffile as tiff
import Utils
import matplotlib.pyplot as plt
import UnetTrain
import split_merge_tif

def to_var(x):
    if torch.cuda.is_available():
        return Variable(x.cuda())
    else:
        return Variable(x)

def NetValidation(net,val_data,path = "data/results/"):
    for i, temp in enumerate(val_data, 0):
        data_, mask_ = temp
        # if i > 0: return 1        # for debug
        data = to_var(data_)
        mask = to_var(mask_)
        out = net(data)

        # print "data: ", data
        # print "out: ", out
        # print "hidden : ", hidden

        img = out.cpu()
        img = img.data.numpy()
        img = img[0, 0, :, :]

        imgname = path + "debug" + ".tif"
        img_shape = img.shape
        img = img.reshape(-1)
        img[img < 0.5] = 0; img[img >= 0.5] = 1
        img = img.reshape(img_shape)
        img = np.uint8(img * 255)
        # tiff.imsave(imgname, img)

        data_ = data_[0,0,62:574,62:574]
        mask_ = mask_[0, 0, :, :]

        '''For Validation data analysis
        plt.subplot(131)
        plt.imshow(data_.numpy(), cmap='gray')
        plt.title("image " + str(i))
        plt.subplot(132)
        plt.imshow(mask_.numpy(), cmap='gray')
        plt.title("Groundtruth " + str(i))
        plt.subplot(133)
        plt.imshow(img, cmap='gray')
        plt.title("Prediction " + str(i))
        plt.show()
        '''

        '''for Test data'''
        plt.subplot(121)
        plt.imshow(data_.numpy(), cmap='gray')
        plt.title("image " + str(i))
        plt.subplot(122)
        plt.imshow(img, cmap='gray')
        plt.title("Prediction " + str(i))
        plt.show()

def create_val_mask_error_metrics(net, val_dataloader, path = "data/results/val_error_metrics/", leg = "Leg0"):
    imgarr = []
    maskarr = []
    inputarr = []
    count = 0
    for i, temp in enumerate(val_dataloader, 0):
        data, mask = temp
        if count >= 30: break
        mask = mask[0, 0, :, :]
        mask = mask.numpy()
        mask = np.uint8(mask * 255)
        # mask_name = path + str(count) + "_mask.tif"
        # tiff.imsave(mask_name, mask)

        if np.sum(mask) < 0.01 * np.shape(mask)[0] * np.shape(mask)[1]: continue
        img = UnetTrain.NetPredict(net, data)
        img = np.uint8(img * 255)
        # img_name = path + str(count) + "_img.tif"
        # tiff.imsave(img_name, img)

        data = data.numpy()
        data = np.uint8(data*255)
        inputarr.append(data)
        imgarr.append(img)
        maskarr.append(mask)
        count += 1

    tiff.imsave(path + "val_input_30_" + leg + ".tif", np.array(inputarr))
    tiff.imsave(path + "val_prediction_30_" + leg + ".tif", np.array(imgarr))
    tiff.imsave(path + "val_mask_30_" + leg + ".tif", np.array(maskarr))

def PlotLoss(model_file, loss_train_path, loss_val_path, epoch_No):
    # analyze loss function of validation set
    loss_train = np.load(loss_train_path)
    loss_val = np.load(loss_val_path)
    plt.figure(1)
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.legend(["Training loss", "validation loss"])
    plt.xticks(np.arange(epoch_No, step=1))
    plt.title(model_file)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    file_name = "loss_graph_" + model_file + ".png"
    plt.savefig(file_name)


def analysis(model_file, valloader):
    path = "data/dataset"
    imgs_test = Utils.load_test_data(npy_path=path)
    testloader = torch.utils.data.DataLoader(imgs_test, batch_size=1, shuffle=False, num_workers=2)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # net = Unet.UNet()  # Unet 2 with batchNorm
        net = Unet_model.UNet_BN()
        # net = Unet_ResNet.UNet_ResNet()
        net.load_state_dict(torch.load(model_file))
        # net.eval()
        net.cuda()

    # Test saved model
    if(False):
        create_val_mask_error_metrics(net, valloader, leg = model_file)
        precision, recall, F1_measure, iou = UnetTrain.Net_performace_val(net, valloader)
        print(model_file)
        disp = "Precision: %.5f \nRecall: %.5f \nF1_measure: %.5f \niou: %.5f" % (precision, recall, F1_measure, iou)
        print(disp)
        UnetTrain.write_2_log(model_file)
        UnetTrain.write_2_log(disp)

    # for TEST data. display test prediction with diff model saved
    if (True):
        UnetTrain.NetTest(net, testloader, path="data/results/")
        test_result_file = "Test_submission" + model_file + ".tif"
        split_merge_tif.merge_img(path="data/results/", merged_name=test_result_file)

        for i in range(2):
            index  = np.random.randint(30)
            mask = tiff.imread("data/results/" + str(index) + ".tif")  # dimension 30x512x512
            img = tiff.imread("data/test/" + str(index) + ".tif")  # dimension 30x512x512
            plt.figure(2)
            plt.subplot(121)
            plt.imshow(img, cmap='gray')
            plt.title("Test_Prediction_" + model_file)
            plt.subplot(122)
            plt.imshow(mask, cmap='gray')
            plt.title("Prediction")
            # plt.show()
            plt.savefig("Test_Prediction_" + model_file + ".png"); break

# analysis()


