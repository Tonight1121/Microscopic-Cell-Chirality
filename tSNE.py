import torch.nn as nn
import torch
import math
import cv2
import torch.utils.model_zoo as model_zoo
import os
import tensorflow as tf
import time
from torch.autograd import Variable
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib import *
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import figure

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from os import path
import tensorflow as tf
#from skimage import io
from PIL import Image

import numpy as np
from PIL import ImageDraw
# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import SubplotTool, Button, Slider, Widget
from scipy.cluster.vq import vq, kmeans, whiten

from matplotlib.backends import pylab_setup
#############################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=1,            # n_filters
                kernel_size=(24, 3),              # filter size
                stride=1,                   # filter movement/step
                padding=0,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            # nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=1,  # n_filters
                kernel_size=(24, 3),  # filter size
                stride=1,  # filter movement/step
                padding=0,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv3 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        self.fc = nn.Linear(24, 3)
    def forward(self, x):
        print(x)
        # time.sleep(10)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc
        output = self.out(x)
        return output, x

class FC(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc = nn.Linear(24, 24)
        self.fc_out = nn.Linear(24, 3)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc(x)
        x = self.fc(x)
        x = self.fc_out(x)
        return x

# load txt labels
# coordinates, cw/ccw/complex corresponding to blue/green/red boxes
def load_labels(set_index):
    img = cv2.imread('data/10x_XY0{}_video_8bit.png'.format(set_index), 1)
    # draw = ImageDraw.Draw(img)
    input_labels = 'data/labels0{}.txt'.format(set_index)
    with open(input_labels, 'r') as myfile:
        data = myfile.read().replace('\n', ' ')
    data = data.split()
    print(data)
    print(len(data))
    for i in range(0, len(data)):
        if data[i] == 'CW':
            data[i] = 0
        elif data[i] == 'CCW':
            data[i] = 1
        elif data[i] == 'Complex':
            data[i] = 2
        elif data[i] == 'NR':
            data[i] = 3
        else:
            data[i] = int(data[i])
    print('data after translation\n{}'.format(data))
    data_array = np.asarray(data)
    print(data_array)
    data_array = np.reshape(data_array, (int(len(data_array) / 3), 3))
    # print(data)
    # print(data_array.shape)
    print(data_array)
    print('data array shape {}'.format(data_array.shape))

    w, h = 50, 50
    for index, cell in enumerate(data_array):
        x, y = cell[0] - 25, cell[1] - 25
        if cell[2] == 0:
            color = (255, 0, 0)
        elif cell[2] == 1:
            color = (0, 255, 0)
        elif cell[2] == 2:
            color = (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, '{}'.format(index),
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    color=color)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.imshow('frame1', img)
    cv2.imwrite('rotations.jpg', img)
    return data_array

# train the network using different modules
def train():
    model = FC().double()
    model = model.to(device)
    # criterion = nn.L1Loss().double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    hist_array = load_data(10, get_avg=True)
    hist_array = np.reshape(hist_array, (1, 24))
    hist_tensor = torch.from_numpy(hist_array).to(device)
    hist_tensor = hist_tensor

    # label_tensor = torch.Tensor([[1, 0, 0]])
    label_tensor = torch.Tensor([0])
    label_tensor = label_tensor.to(device)

    for epoch in range(0, 50):
        outputs = model(hist_tensor)
        _, preds = torch.max(outputs, 1)
        predict = nn.Softmax()
        scores = predict(outputs)
        # print(scores)
        # time.sleep(50)
        # print('score {}'.format(scores))
        # print('label {}'.format(label_tensor))
        # label_tensor.requires_grad = True
        # print(scores.requires_grad)
        # print(label_tensor.requires_grad)

        scores = scores.type(torch.DoubleTensor)
        label_tensor = label_tensor.type(torch.LongTensor)
        loss = criterion(scores, label_tensor)
        print('ep {}/50 - loss {:.2f}'.format(epoch, loss))
        loss.backward()
        optimizer.step()

# load 24bin data generated from Main10...
# get the average or not
def load_data(index, get_avg=False):
    hist_bin = np.loadtxt('hist/cell_{}.txt'.format(index))
    # print(len(hist_bin))
    hist_bin = np.reshape(hist_bin, (int(len(hist_bin)/24), 24))
    temp = hist_bin[0]
    if get_avg == True:
        for i in range(1, hist_bin.shape[0]):
            temp = temp + hist_bin[i]
        hist_bin = temp / hist_bin.shape[0]
        hist_bin = np.reshape(hist_bin, (1, 24))
    print('{} hist_bin shape {}'.format(index, hist_bin.shape))
    return hist_bin

def dimension_reduction(method, dimension):
    my_arrays = load_data(0, get_avg=True)
    for i in range(1, 89):
        my_arrays = np.concatenate((my_arrays, load_data(i, get_avg=True)), axis=0)
    # print(my_arrays.shape)
    my_tensor = torch.from_numpy(my_arrays)
    my_labels = np.loadtxt('hist/rotation.txt')
    print(my_labels.shape)
    # time.sleep(30)
    color_choice = ['b', 'g', 'r', 'yellow']
    # print(my_tensor)
    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='random',
                             random_state=500,
                             early_exaggeration=5,
                             method='exact')
    elif method == 'PCA':
        module = PCA(n_components=3)
    x_numpy = my_tensor.data.cpu().numpy()
    x_reduced = module.fit_transform(x_numpy)
    print(x_reduced.shape)
    print(x_reduced)
    plt.figure()
    if dimension == 2:
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            plt.plot(x_reduced[i, 0], x_reduced[i, 1], 'o', color=color_choice[color_index])
        plt.xticks()
        plt.yticks()

    elif dimension == 3:
        ax = plt.axes(projection='3d')
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            ax.scatter(x_reduced[i, 0], x_reduced[i, 1], x_reduced[i, 2], color=color_choice[color_index])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    plt.savefig('plots/{}-24bin-{}d.pdf'.format(method, dimension))
    plt.show()



def feature_PCA():
    my_arrays = load_data(0, get_avg=True)
    for i in range(1, 89):
        my_arrays = np.concatenate((my_arrays, load_data(i, get_avg=True)), axis=0)
    my_tensor = torch.from_numpy(my_arrays)
    my_labels = np.loadtxt('hist/rotation.txt')
    color_choice = ['b', 'g', 'r', 'yellow']
    pca = PCA(n_components=3)
    x_numpy = my_tensor.data.cpu().numpy()
    x_pca = pca.fit_transform(x_numpy)

    plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(x_pca.shape[0]):
        color_index = int(my_labels[i])
        # plt.plot(x_pca[i, 0], x_pca[i, 1], 'o', color=color_choice[color_index])
        ax.scatter(x_pca[i, 0], x_pca[i, 1], x_pca[i, 2], color=color_choice[color_index])
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
        #          fontdict={'weight': 'bold', 'size': 9})
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # plt.xlim((-200, 0))
    # plt.ylim((-100, 100))
    # plt.xticks()
    # plt.yticks()
    # plt.savefig('tSNE-24bin.pdf')
    # plt.show()

def feature_visualization(method, dimension):
    color_choice = ['b', 'g', 'r', 'yellow']
    x_numpy = np.loadtxt('experiments/features.txt')
    my_labels = np.loadtxt('experiments/features_labels.txt')
    print(x_numpy.shape)
    time.sleep(30)
    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='pca',
                             random_state=500,
                             early_exaggeration=20,
                             method='exact')
    elif method == 'PCA':
        module = PCA(n_components=3)
    x_reduced = module.fit_transform(x_numpy)
    print(x_reduced.shape)
    print(x_reduced)
    plt.figure()
    if dimension == 2:
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            if my_labels[i] == 0:
                status = 'Died'
            else:
                status = 'Survived'
            plt.plot(x_reduced[i, 0], x_reduced[i, 1], 'o', color=color_choice[color_index], label=status)
        plt.xticks()
        plt.yticks()

    elif dimension == 3:
        ax = plt.axes(projection='3d')
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            ax.scatter(x_reduced[i, 0], x_reduced[i, 1], x_reduced[i, 2], color=color_choice[color_index])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    plt.legend(loc="lower right")
    plt.savefig('experiments/{}_{}.pdf'.format(method, dimension))
    plt.show()

def feature_visualization2(method, dimension, point_num=1700):
    color_choice = ['b', 'g', 'r', 'yellow']
    x_numpy = np.loadtxt('data/gt_vectors.txt')
    my_labels = np.loadtxt('data/gt_labels.txt')
    print('my_labels {}'.format(my_labels))

    my_labels = np.reshape(my_labels, (my_labels.shape[0], 1))
    mydata = np.concatenate((my_labels, x_numpy), axis=1)
    np.random.shuffle(mydata)
    x_numpy = mydata[:, 1:]
    my_labels = mydata[:, 0]
    x_numpy = x_numpy[:point_num, :]
    my_labels = my_labels[:point_num]

    print('Size of vectors {}'.format(x_numpy.shape))
    print('Size of labels {}'.format(my_labels.shape))
    # print('labels {}'.format(my_labels))
    died_index = np.where(my_labels == 0)
    survived_index = np.where(my_labels == 1)
    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='pca',
                             random_state=10,
                             early_exaggeration=20,
                             method='barnes_hut')
    elif method == 'pca':
        module = PCA(n_components=100)
    x_reduced = module.fit_transform(x_numpy)
    died_vector = x_reduced[died_index]
    survived_vector = x_reduced[survived_index]
    plt.figure()
    if dimension == 2:
        plt.plot(died_vector[:, 0], died_vector[:, 1], 'o', color='tomato', alpha=0.5, label='Deceased')
        plt.plot(survived_vector[:, 0], survived_vector[:, 1], 'o', color='limegreen', alpha=0.5, label='Survived')
        # plt.plot(x_reduced[i, 0], x_reduced[i, 1], 'o', color=color_choice[color_index])
        plt.xticks([])
        plt.yticks([])

    elif dimension == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(died_vector[:, 0], died_vector[:, 1], died_vector[:, 2], color='r', label='Died')
        ax.scatter(survived_vector[:, 0], survived_vector[:, 1], survived_vector[:, 2], color='g', label='Survived')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    # plt.title('t-SNE Visualization of Patch-34 Feature Vectors')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig('experiments/{}_{}.pdf'.format(method, dimension))
    plt.savefig('experiments/{}_{}.png'.format(method, dimension))
    plt.show()

def feature_visualization3(method, dimension, fold):
    color_choice = ['b', 'g', 'r', 'yellow']
    x_numpy = np.loadtxt('experiments/features.txt')
    x_numpy = x_numpy[54*fold: 54*(fold+1)]
    my_labels = np.loadtxt('experiments/features_labels.txt')
    my_labels = my_labels[54*fold: 54*(fold+1)]
    print('Size of labels {}'.format(my_labels.shape))
    # print('labels {}'.format(my_labels))
    died_index = np.where(my_labels == 0)
    survived_index = np.where(my_labels == 1)
    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='pca',
                             random_state=100,
                             early_exaggeration=2000,
                             method='exact')
    elif method == 'PCA':
        module = PCA(n_components=10)
    x_reduced = module.fit_transform(x_numpy)
    died_vector = x_reduced[died_index]
    survived_vector = x_reduced[survived_index]
    print(x_reduced)
    # plt.subplot(2, 5, i+1, figsize=(15, 15))
    plt.figure(figsize=(5,5))
    if dimension == 2:
        plt.plot(died_vector[:, 0], died_vector[:, 1], 'o', color='r', alpha=0.5, label='Deceased')
        plt.plot(survived_vector[:, 0], survived_vector[:, 1], 'o', color='g', alpha=0.5, label='Survived')
        # axs[i//5, i%5].plot(died_vector[:, 0], died_vector[:, 1], 'o', color='r', alpha=0.5, label='Died')
        # axs[i//5, i%5].plot(survived_vector[:, 0], survived_vector[:, 1], 'o', color='g', alpha=0.5, label='Survived')
        # axs[i // 5, i % 5].axis('off')
        # axs[i // 5, i % 5].set_xticks([])
        # axs[i // 5, i % 5].set_yticks([])
        # plt.plot(x_reduced[i, 0], x_reduced[i, 1], 'o', color=color_choice[color_index])
        plt.xticks([])
        plt.yticks([])

    elif dimension == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(died_vector[:, 0], died_vector[:, 1], died_vector[:, 2], color='r', label='Died')
        ax.scatter(survived_vector[:, 0], survived_vector[:, 1], survived_vector[:, 2], color='g', label='Survived')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    # plt.title('t-SNE Visualization of Patch-34 Feature Vectors')
    plt.legend(loc="lower right")
    plt.savefig('experiments/tsne/{}_{}_{}.pdf'.format(method, dimension, fold))
    plt.savefig('experiments/tsne/{}_{}_{}.jpg'.format(method, dimension, fold))
    plt.show()
    # plt.subplots(2, 5, 10, i)
    # return plt

def feature_visualization_remake(vectors, labels, method, dimension, figname):
    class_num = int(np.max(labels) + 1)
    print('there are {} classes'.format(class_num))
    # time.sleep(30)
    color_choice = ['g', 'r', 'b', 'yellow']
    x_numpy = vectors
    my_labels = labels
    print('my_labels {}'.format(my_labels))

    my_labels = np.reshape(my_labels, (my_labels.shape[0], 1))
    mydata = np.concatenate((my_labels, x_numpy), axis=1)
    np.random.shuffle(mydata)
    x_numpy = mydata[:, 1:]
    my_labels = mydata[:, 0]

    print('Size of vectors {}'.format(x_numpy.shape))
    print('Size of labels {}'.format(my_labels.shape))

    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='random',
                             random_state=10,
                             early_exaggeration=20,
                             method='barnes_hut')
    elif method == 'pca':
        module = PCA(n_components=100)
    x_reduced = module.fit_transform(x_numpy)

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.tight_layout()

    if dimension == 2:
        for class_index in range(0, class_num):
            this_index = np.where(my_labels == class_index)
            class_vectors = x_reduced[this_index]
            plt.plot(class_vectors[:, 0], class_vectors[:, 1], 'o',
                     color=color_choice[class_index], alpha=0.5, label='Class {}'.format(class_index))
            if class_index == 1:
                plt.legend(loc="upper right")
                plt.savefig('experiments/{}_{}_{}_2.png'.format(method, dimension, figname))


    elif dimension == 3:
        ax = plt.axes(projection='3d')
        for class_index in range(0, class_num):
            this_index = np.where(my_labels == class_index)
            class_vectors = x_reduced[this_index]
            ax.scatter(class_vectors[:, 0], class_vectors[:, 1], class_vectors[:, 2],
                       color=color_choice[class_index], label='Class {}'.format(class_index))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    # plt.title('t-SNE Visualization of Patch-34 Feature Vectors')
    # plt.legend(loc="upper right")
    # plt.savefig('experiments/{}_{}_{}.pdf'.format(method, dimension))
    plt.savefig('experiments/{}_{}_{}_3.png'.format(method, dimension, figname))
    # plt.show()

def equal_class(composed_data, num0=100, num1=100, num2=100):
    np.random.shuffle(composed)
    labels = composed_data[:, 24]

    index_0 = np.where(labels == 0)
    index_1 = np.where(labels == 1)
    index_2 = np.where(labels == 2)

    data_0 = composed_data[index_0][:num0, :]
    data_1 = composed_data[index_1][:num1, :]
    data_2 = composed_data[index_2][:num2, :]

    print('1 {}, 2 {}, 3 {}'.format(data_0.shape, data_1.shape, data_2.shape))
    processed = np.concatenate((data_0, data_1), axis=0)
    processed = np.concatenate((processed, data_2), axis=0)

    print('processed shape {}'.format(processed.shape))
    return processed



if __name__ == '__main__':

    # feature_vectors = np.loadtxt('data/gt_vectors.txt')
    # gt_labels = np.loadtxt('data/gt_labels.txt')
    feature_vectors = np.loadtxt('data/allvec.txt')
    gt_labels = np.loadtxt('data/all1645.txt')
    print('feature vector shape {}'.format(feature_vectors.shape))
    print('gt_labels shape {}'.format(gt_labels.shape))

    gt_labels = np.reshape(gt_labels, (gt_labels.shape[0], 1))
    composed = np.concatenate((feature_vectors, gt_labels), axis=1)

    cw_num = np.count_nonzero(gt_labels == 0)
    ccw_num = np.count_nonzero(gt_labels == 1)
    nr_num = np.count_nonzero(gt_labels == 2)
    print('cw {}, ccw {}, nr {}'.format(cw_num, ccw_num, nr_num))
    # time.sleep(30)

    # composed = equal_class(composed)

    iteration = 1
    for i in range(0, iteration):
        np.random.shuffle(composed)
        feature_vectors = composed[:, :24]
        gt_labels = composed[:, 24]

        feature_visualization_remake(vectors=feature_vectors,
                                     labels=gt_labels,
                                     method='tsne',
                                     dimension=2,
                                     figname=i)














    # cv2.waitKey(0)