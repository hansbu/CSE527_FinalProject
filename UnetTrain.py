from __future__ import print_function, division
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Unet_model
import tifffile as tiff
import Utils
import time
from sklearn.metrics import average_precision_score, recall_score
import Analysis

def to_var(x):
    if torch.cuda.is_available():
        return Variable(x.cuda())
    else:
        return Variable(x)

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

    def forward(self, probs, targets):
        probs = F.sigmoid(probs)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)      # (output, labels)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(probs)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

def _criterion(logits, labels):
    return BinaryCrossEntropyLoss2d().forward(logits, labels) + \
        SoftDiceLoss().forward(logits, labels)

def NetEval(net,valdata_loader,crit):
    loss = 0.
    count = 0
    net.eval()
    for i,data in enumerate(valdata_loader,0):
        inputs, labels = data
        inputs, labels = to_var(inputs), to_var(labels)
        out = net(inputs)
        l = crit(out, labels)
        loss += l.data[0]
        count += 1
    return loss/count

# calculate Precision, Recall, F1-measure, IoU over the validation set
def Net_performace_val(net,valdata_loader):
    precision_list = []
    recall_list = []
    iou_list = []
    net.eval()
    print("len of valdata: ", len(valdata_loader))
    for i,data in enumerate(valdata_loader,0):

        inputs, labels = data
        out = NetPredict(net, inputs)   # out is numpy array 512x512
        labels = labels.numpy()
        labels = labels[0,0,:,:]        # labels is numpy array 512x512
        # remove the case where the mask is plain black
        if np.sum(labels) < 0.01*np.shape(labels)[0]*np.shape(labels)[1]: continue

        precision = average_precision_score(labels, out)
        recall = recall_score(labels, out, average='weighted')
        iou = IoU(out, labels)
        precision_list.append(precision)
        recall_list.append(recall)
        iou_list.append(iou)

    precision, recall, iou = np.mean(precision_list), np.mean(recall_list), np.mean(iou_list)
    F1_measure = 2.0*precision*recall/(precision + recall)

    return precision, recall, F1_measure, iou

# calculate IoU
def IoU(pred, target):
    # Ignore IoU for background class
    cls = 0.5
    pred_inds = pred > cls
    target_inds = target > cls

    intersection = (pred_inds[target_inds]).sum()  # Cast to long to prevent overflows
    union = pred_inds.sum() + target_inds.sum() - intersection
    if union == 0:
        ious = float('nan')  # If there is no ground truth, do not include in evaluation
    else:
        ious = 1.0*intersection / max(union, 1)
    return ious

#process the test data and save to separate .tif image
def NetTest(net,testdata_loader,path = "data/results/"):
    net.eval()
    for i, data in enumerate(testdata_loader, 0):
        if i >= 30: return 1        # for debug
        data = to_var(data)
        out = net(data)

        out = F.sigmoid(out)

        imgname = path + str(i) + ".tif"
        img = out.cpu()
        img = img[0,0,:,:]
        img = img.data.numpy()
        img_shape = img.shape
        img = img.reshape(-1)
        img[img < 0.5] = 0; img[img >= 0.5] = 1
        img = img.reshape(img_shape)
        img = np.uint8(img * 255)
        tiff.imsave(imgname, img)

# return input image size like 512x512, numpy array
def NetPredict(net, data):
    net.eval()
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
    return img


def write_2_log(content):
    path = "log_file.txt"
    output = open(path, "a")
    output.write("\n")
    output.write(content)
    output.close()

if __name__ == '__main__':
    '''subjects to change for each exp:
    + path to the dataset
    + opt function
    + leg description
    + N_epoch
    + saved_model
    '''
    start_begining = time.time()

    # path = "data/dataset"
    path = "/home/anhxtuan/HanLe/dataset/20aug_10ela_3inten_reflect"

    import os
    print("Does the folder exist?: ", os.path.isdir(path))
    print("Does the path exist?: ", os.path.exists(path))

    imgs_train, imgs_mask_train = Utils.load_train_data(npy_path=path)
    imgs_val, imgs_mask_val = Utils.load_val_data(npy_path=path)
    batch_size = 1
    trainset = zip(imgs_train, imgs_mask_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    valset = zip(imgs_val, imgs_mask_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    # net = Unet.UNet()                     # Unet 1
    net = Unet_model.UNet_BN()          # Unet 2 with batchNorm

    f1 = open("Unet_model.py", 'r')     # embede code into the model to be saved later
    f2 = open("UnetTrain.py", 'r')
    f3 = open("Utils.py", 'r')
    model_notes = time.strftime("%c") + "\n" + f1.read() + "\n" + "==="*5 + f2.read() + "\n" + "==="*5 + f3.read()
    f1.close()
    f2.close()
    f3.close()

    criterion = BinaryCrossEntropyLoss2d()
    # criterion = SoftDiceLoss()
    # criterion = nn.BCELoss()

    if torch.cuda.is_available():
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # net.load_state_dict(torch.load("Unet2_model_12epoch_Leg12_dropout_0_2_rmSigmoid_20aug_10ela_3inten_reflect"))
        net.cuda()
        # criterion.cuda()

    opt = optim.SGD(net.parameters(), lr=1e-5, momentum=0.8)       # subject to change
    # opt = optim.RMSprop(net.parameters(), lr=2e-4)
    # RMSprop[22](learning rate 0.001) with weight decay set to 0.001.
    # opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loss = []
    val_loss = []

    print("[epoch, #image] Loss")
    leg = "Leg21_finetuneLeg12_dropout_0_2_rmSigmoid_20aug_10ela_3inten_reflect"
    print("Processing: ", leg)
    model_name = ""

    N_epoch = 6
    model_saved = np.arange(2,N_epoch+1,2)
    val_loss_min = 100

    write_2_log("==="*30)
    write_2_log(time.strftime("%c"))       # time, date of exp
    write_2_log(leg)
    write_2_log(model_notes)
    write_2_log(time.strftime("%c"))  # time, date of exp
    write_2_log(leg)
    write_2_log("optim.SGD(net.parameters(), lr=1e-5, momentum=0.8)")
    write_2_log("Number of epoch max: " + str(N_epoch))


    min_model_epoch = 0
    for epoch in range(N_epoch):
        running_loss = 0.0; count = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            temp = labels.numpy()
            temp = temp[0, 0, :, :]  # labels is numpy array 512x512
            # remove the case where the mask is plain black
            if np.sum(temp) < 0.01 * np.shape(temp)[0] * np.shape(temp)[1]: continue
            inputs, labels = to_var(inputs), to_var(labels)
            opt.zero_grad()
            out = net(inputs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.data[0]
            count += 1

        net.eval()
        val_loss_temp = NetEval(net, valloader, criterion)
        if epoch > 0 and val_loss_temp < val_loss_min:        # save the model which has the lowest val_loss, ignore first epoch
            val_loss_min = val_loss_temp
            torch.save(net.state_dict(), "Unet2_model_" + leg + "_min_val_loss")  # save model
            min_model_epoch = epoch + 1

        disp = 'Epoch %d: Train loss: %.6f, Val loss: %.6f' % (epoch + 1, running_loss/count, val_loss_temp)
        write_2_log(disp)
        print(disp)
        val_loss.append(val_loss_temp)
        train_loss.append(running_loss/count); running_loss = 0.0

        if (epoch + 1) in model_saved:        # save checkpoint
            model_name = "Unet2_model_" + str(epoch+1) + "epoch_" + leg
            torch.save(net.state_dict(), model_name)  # save model
            Analysis.analysis(model_file = model_name, valloader = valloader)

    write_2_log("Best model at epoch: " + str(min_model_epoch))
    write_2_log("Total training time: " + str((time.time() - start_begining) / 3600.0))

    val_loss_file = "data/results" + "/val_loss_" + str(N_epoch) + "epoch_" + leg + ".npy"
    train_loss_file = "data/results" + "/training_loss_" + str(N_epoch) + "epoch_" + leg + ".npy"

    np.save(val_loss_file, val_loss)
    np.save(train_loss_file, train_loss)
    print("total processing time in seconds: ", time.time() - start_begining)
    print("total processing time in mins: ", (time.time() - start_begining) / 60.0)
    print("total processing time in hours: ", (time.time() - start_begining) / 3600.0)

    # Analysis.analysis(model_file="Unet2_model_" + leg + "_min_val_loss", valloader=valloader)

    Analysis.PlotLoss(model_file = model_name, loss_train_path = train_loss_file, loss_val_path = val_loss_file, epoch_No = N_epoch)

    write_2_log(time.strftime("%c"))  # time, date of exp