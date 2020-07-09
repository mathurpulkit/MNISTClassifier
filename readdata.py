import numpy as np
import idx2numpy
import matplotlib.pyplot as plt


def read_input(ifilename = "train-images.idx3-ubyte", lfilename="train-labels.idx1-ubyte"): #reads img file and label files and returns arrays
    images = idx2numpy.convert_from_file(ifilename) #variable will store images in 3-D array
    imgdata = np.reshape(images,newshape=[images.shape[0],-1])/255
    #image data reshaped to 2-D array for NN
    f = open(lfilename, 'rb')
    f.read(4)
    size = int.from_bytes(f.read(4), 'big')
    labels = np.reshape(np.frombuffer(f.read(), dtype=np.uint8), (-1, 1))
    labeldata = process_labeldata(labels)
    imgdata = imgdata.T #turns into no_of_pix*m array
    #1-D array for labeldata
    #pic = np.asarray(images[20].squeeze())
    #plt.imshow(pic)
    #plt.show()
    return imgdata, labeldata, images


def process_labeldata(x):
    labeldata = np.zeros((10, x.shape[0]))
    f = open("dump.txt", "w")
    for i in range(labeldata.shape[1]):
        labeldata[x[i][0]][i] = 1
    return labeldata
