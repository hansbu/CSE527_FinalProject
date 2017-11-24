'''
split 30 single images from an array of images : train-volume.tif label-volume.tif test-volume.tif
'''
import tifffile as tiff
import numpy as np

dirtype = ("train", "label", "test")

def split_img(path = "data/raw/" ):
    '''
	split a tif volume into single tif
	'''

    for t in dirtype:
        imgarr = tiff.imread(path + t + "-volume.tif")  # dimension 30x512x512
        for i in range(imgarr.shape[0]):
            imgname = "data/" + t + "/" + str(i) + ".tif"
            tiff.imsave(imgname, imgarr[i])

def merge_img(path = "data/results/", merged_name = "result_3epoch.tif"):
    '''
	merge single tif into a tif volume
	'''
    imgarr = []
    for i in range(30):
        img = tiff.imread(path + str(i) + ".tif")
        imgarr.append(img)

    tiff.imsave(path + merged_name, np.array(imgarr))


if __name__ == "__main__":

    # split_img()
    merge_img(path = "data/results/", merged_name = "result_3epoch.tif")
