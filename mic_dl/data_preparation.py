import urllib.request
import cv2
import numpy as np
import os
from os import path
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import resnet_regression as resnet
import pickle
import sklearn
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import copy
import random

data_dir = path.expanduser('/zion/guoh9/mic_data/data')

net = 'ResNet-34'

datafolder = path.expanduser('/zion/guoh9/mic_data/patches_104x104/patches_104x104/')
files = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if net == 'ResNet-18':
    base_model = resnet.resnet18
elif net == 'ResNet-34':
    base_model = resnet.resnet34
elif net == 'ResNet-50':
    base_model = resnet.resnet50
elif net == 'ResNet-101':
    base_model = resnet.resnet101
elif net == 'ResNet-152':
    base_model = resnet.resnet152
else:
    print('The network of {} is not supported!'.format(net))

model = base_model(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
fn_best_model = os.path.join(data_dir, 'best_scratch_{}.pth'.format(net))
model.load_state_dict(torch.load(fn_best_model))
model.eval()
model = model.to(device)

# Just normalization for validation
data_transform = transforms.Compose([
        transforms.CenterCrop(52),
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(vec_mean, vec_std)
    ])

def get_batch_name(data_path, batch_size=32):
    files = os.listdir(data_path)
    files.sort()

    train_num = int(0.8 * len(files))
    val_num = int(0.1 * len(files))

    train_set = files[:train_num]
    val_set = files[train_num:train_num + val_num]
    test_set = files[train_num + val_num:]
    print('trainset {}, valset {}, testset {}'.format(len(train_set), len(val_set), len(test_set)))

    train_batches = [train_set[batch_size * i:batch_size * (i + 1)] for i in range(int(len(train_set) / 4 + 1))]
    train_batches = [x for x in train_batches if x != []]

    val_batches = [val_set[batch_size * i:batch_size * (i + 1)] for i in range(int(len(val_set) / 4 + 1))]
    val_batches = [x for x in val_batches if x != []]

    test_batches = [test_set[batch_size * i:batch_size * (i + 1)] for i in range(int(len(test_set) / 4 + 1))]
    test_batches = [x for x in test_batches if x != []]

    return train_batches, val_batches, test_batches

def batch2tensor(batch_name):
    batch_size = len(batch_name)

    img = np.zeros()

    for img_name in batch_name:
        img_path = path.join(datafolder, img_name)
        print(img_path)
    return 0, 0

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

''' Input two 52*52 interframe patches, output network prediction result'''
def interframe_pred(patch_1, patch_2):
    patch_1 = Image.fromarray(np.uint8(patch_1 * 255), 'L')
    patch_2 = Image.fromarray(np.uint8(patch_2 * 255), 'L')

    ''' Data transform, resize to 224, to tensor'''
    patch_1 = data_transform(patch_1)
    patch_2 = data_transform(patch_2)

    ''' Concatenate two image into 2*224*224 '''
    image = torch.cat((patch_1, patch_2), 0)

    ''' To GPU, view in 1*2*224*224 to apply to the network'''
    image = image.to(device)
    image = image.view(1, image.shape[0], image.shape[1], image.shape[2])

    ''' Network result, get prediction '''
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    preds = preds.data.cpu().numpy()[0]

    print('outputs {}'.format(outputs))

    return preds

def predict_cell(file_name):
    # cell = np.load('/zion/guoh9/mic_conv3d/req_npy/train/cw/{}'.format(file_name))
    file_path = path.join(folder, file_name)
    cell = np.load(file_path)
    # cell = np.load('/zion/guoh9/mic_conv3d/req_npy/train/ccw/0118.npy')

    pred_np = np.zeros((24,))
    print(pred_np)

    for i in range(0, cell.shape[0] - 1):
        patch_former = cell[i, :, :]
        patch_after = cell[i + 1, :, :]
        pred = interframe_pred(patch_former, patch_after)
        pred_np[i] = pred
        # print('prediction: {}'.format(pred))

    unique_elements, counts_elements = np.unique(pred_np, return_counts=True)
    print('unique: {}, counts: {}'.format(unique_elements, counts_elements))
    # print('counts: {}'.format(counts_elements))


if __name__ == '__main__':
    rands = np.random.random_integers(low=-5, high=5, size=(1645))

    rands_positive = np.random.random_integers(low=1, high=5, size=(822,))
    rands_negative = np.random.random_integers(low=-5, high=-1, size=(823,))
    zeros = np.zeros((549,))

    rands = np.concatenate((rands_positive, rands_negative), axis=0)
    # rands = np.concatenate((rands, zeros), axis=0)
    np.random.shuffle(rands)
    print('rands {}'.format(rands))
    print('rands shape {}'.format(rands.shape))
    # time.sleep(30)

    np.savetxt('rand_angle.txt', rands)
    print(rands)
    time.sleep(30)

    # angles = np.loadtxt('rand_angle.txt')
    # print(angles)

    ''' ====================================================== '''
    # cell = np.load('/zion/guoh9/mic_conv3d/req_npy/train/cw/0499.npy')
    # # cell = np.load('/zion/guoh9/mic_conv3d/req_npy/train/ccw/0118.npy')
    #
    # pred_np = np.zeros((24,))
    # print(pred_np)
    #
    # for i in range(0, cell.shape[0]-1):
    #     patch_former = cell[i, :, :]
    #     patch_after = cell[i+1, :, :]
    #     pred = interframe_pred(patch_former, patch_after)
    #     pred_np[i] = pred
    #     # print('prediction: {}'.format(pred))
    #
    # unique_elements, counts_elements = np.unique(pred_np, return_counts=True)

    ''' ======================================================== '''
    # folder = '/zion/guoh9/mic_conv3d/req_npy/train/ccw/'
    # for file_name in os.listdir(folder):
    #     predict_cell(file_name)

    frame1 = '00'
    # folder = '/zion/guoh9/mic_data/patches_104x104/patches_104x104/{}'.format(frame1)
    folder = '/zion/guoh9/mic_data/data/val/'

    file_name = '1376.npy'
    file_path = path.join(folder, file_name)
    patch_former = np.loadtxt(file_path)
    # angle = random.randint(-5, 5)
    angle = 4
    patch_after = rotateImage(patch_former, angle)
    pred = interframe_pred(patch_former, patch_after)
    print('{}: angle {}, pred {}'.format(file_name, angle, pred))



    # for file_name in os.listdir(folder):
    #     file_path = path.join(folder, file_name)
    #     patch_former = np.loadtxt(file_path)
    #     angle = random.randint(-5, 5)
    #     patch_after = rotateImage(patch_former, angle)
    #     # cv2.imwrite('patch_former.jpg', patch_former)
    #     # cv2.imwrite('patch_after.jpg', patch_after)
    #     # time.sleep(30)
    #     # patch_after = np.loadtxt('/zion/guoh9/mic_data/patches_104x104/patches_104x104/{}/{}.npy'.format(frame2, index))
    #     pred = interframe_pred(patch_former, patch_after)
    #     print('{}: angle {}, pred {}'.format(file_name, angle, pred))








