import urllib.request
import cv2
import imutils
from pylab import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random
from collections import Counter
import time
from os import path
from os import walk
import pickle
import sklearn
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import copy
from sklearn.svm import SVC
# import simple_cnn as sic
# import imutils
# from ssim import SSIM
# from ssim.utils import get_gaussian_kernel
# cw_ssim_rot = SSIM(im).cw_ssim_value(im_rot)
import sys
import time
import SimpleITK as sitk


'''
Zhicheng Guo | Aug 9th, 2019
CHANGE ALL PATHs IN THE FOLLOWING CODE ACCORDINGLY
'''


sys.setrecursionlimit(100)
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, QMessageBox
from PyQt5.QtWidgets import QMenu, QAction, QMainWindow, QLabel, QFileDialog
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit
from PyQt5.QtGui import QPixmap, QFont
import statistics
# from PyQt5 import QtWidgets, QtGui
import time
import cv2
import shutil
from sklearn.metrics import confusion_matrix
from scipy.interpolate import splev, splrep, splprep

# frames = []
# video_id = '05'
# const_loc = np.loadtxt("const_index/loc/{}.txt".format(video_id))
# frame_gap = 1
# ssim_set_val = 0.80
# all_pairs = []
degree_range = 10
step = 0.1
# final_degrees = []
'''
traininng video IDs
'''
test_set_index = [15, 16, 17, 18, 32]
video_fold = path.expanduser('~/Videos')

def load_video_old(frames, frame_gap, video_id):
    """
    Load videos
    :param frames: container for all frames
    :param frame_gap: frame gap
    :param video_id: video name (id)
    :return: None
    """

    '''
    CHANGE TO YOUR OWN VIDEO PATH
    '''
    tmp_frames = []
    video_path = path.join(video_fold, 'XY{}_video.mp4'.format(video_id))
    file_list = os.listdir(video_fold)
    cap = cv2.VideoCapture(video_path)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # cv2.imwrite('new_stuff/frame{}.jpg'.format(count), frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tmp_frames.append(gray)
            # print(count)
            count = count + 1
        else:
            break

    cap.release()

    for i in range(len(tmp_frames)):
        if i % frame_gap == 0:
            frames.append(tmp_frames[i])

    print('VIDEO {} LOADED WITH FRAME COUNT -> {}'.format(video_id, frames.__len__()))

def load_video(frames, frame_gap, video_id):
    """
    Load videos
    :param frames: container for all frames
    :param frame_gap: frame gap
    :param video_id: video name (id)
    :return: None
    """

    '''
    CHANGE TO YOUR OWN VIDEO PATH
    '''
    tmp_frames = []
    video_path = path.join(video_fold, 'vid/XY{}_video'.format(video_id))
    file_list = os.listdir(video_path)
    file_list.sort()

    for img_name in file_list:
        img_path = path.join(video_path, img_name)
        gray = cv2.imread(img_path, 0)
        tmp_frames.append(gray)

    for i in range(len(tmp_frames)):
        if i % frame_gap == 0:
            frames.append(tmp_frames[i])

    print('VIDEO {} LOADED WITH FRAME COUNT -> {}'.format(video_id, frames.__len__()))


def progbar(curr, total, full_progbar):
    """
    Progress bar display in console (helper function)
    :param curr:
    :param total:
    :param full_progbar:
    :return:
    """
    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    print('\r', '#' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='')


def bias_free(vecs, labels, two_class=False, use_flip=True):
    """
    Train a SVM model with equal number of feature vectors from each class
    Option include train a CW&CCW vs Other classifier
                            or
                           CW vs CCW vs Other
    :param vecs: raw feature vectors
    :param labels: raw labels
    :param two_class: True if train for two class, False otherwise
    :param use_flip: if use the fipped feature vectors
    :return: SEE COMMENTS BELOW
    """
    # vecs = np.loadtxt('registration/alltrain_vec.txt')
    # labels = np.loadtxt('registration/alltrain_lab.txt')
    print(vecs.shape, labels.shape)
    cw = []
    ccw = []
    oth = []
    for i in range(labels.shape[0]):
        if labels[i] == 0:
            cw.append(vecs[i])

        elif labels[i] == 1:
            ccw.append(vecs[i])

        else:
            oth.append(vecs[i])

    print(cw.__len__(), ccw.__len__(), oth.__len__())
    # random.seed(5)
    random.shuffle(cw)
    random.shuffle(ccw)
    random.shuffle(oth)

    if two_class:
        selected_cw = random.sample(cw, min(cw.__len__(), ccw.__len__()))
        selected_ccw = random.sample(ccw, min(cw.__len__(), ccw.__len__()))
    else:
        selected_cw = random.sample(cw, min(cw.__len__(), ccw.__len__(), oth.__len__()))
        selected_ccw = random.sample(ccw, min(cw.__len__(), ccw.__len__(), oth.__len__()))
    selected_oth = random.sample(oth, min(cw.__len__(), ccw.__len__(), oth.__len__()))
    print(selected_cw.__len__(), selected_ccw.__len__(), oth.__len__())

    selected_cw = np.asarray(selected_cw)
    print(selected_cw.shape)
    if use_flip:
        flipped_cw = np.flip(selected_cw, axis=1)
        flipped_cw = np.multiply(flipped_cw, -1)

    selected_ccw = np.asarray(selected_ccw)
    if use_flip:
        flipped_ccw = np.flip(selected_ccw, axis=1)
        flipped_ccw = np.multiply(flipped_ccw, -1)

    selected_oth = np.asarray(selected_oth)
    if use_flip:
        flipped_oth = np.flip(selected_oth, axis=1)
        flipped_oth = np.multiply(flipped_oth, -1)

    if use_flip:
        selected_cw = np.concatenate((selected_cw, flipped_ccw))
        selected_ccw = np.concatenate((selected_ccw, flipped_cw))
        selected_oth = np.concatenate((selected_oth, flipped_oth))

    print(selected_cw.shape, selected_ccw.shape, selected_oth.shape)

    all_vec = np.concatenate((selected_cw, selected_ccw))

    if not two_class:
        all_vec = np.concatenate((all_vec, selected_oth))

    cw_label = np.zeros((selected_cw.shape[0],))
    ccw_label = np.zeros((selected_ccw.shape[0],))
    oth_label = np.zeros((selected_oth.shape[0],))
    cw_label.fill(0)
    ccw_label.fill(1)
    oth_label.fill(2)
    all_lab = np.concatenate((cw_label, ccw_label))

    if not two_class:
        all_lab = np.concatenate((all_lab, oth_label))

    print(cw_label.shape, ccw_label.shape, oth_label.shape)

    cw_stdev = []
    ccw_stdev = []
    oth_stdev = []

    for row in selected_cw:
        stdev = np.std(row)
        cw_stdev.append(stdev)

    for row in selected_ccw:
        stdev = np.std(row)
        ccw_stdev.append(stdev)

    for row in selected_oth:
        stdev = np.std(row)
        oth_stdev.append(stdev)
    '''
    COMMENT LINES BELOW FOR TRAINING
    IF NOT, ONLY PROCESSED FEATURE VECTORS AND LABLES WILL BE RETURNED (equal amount for each class)
    '''
    # print(statistics.mean(cw_stdev), statistics.mean(ccw_stdev), statistics.mean(oth_stdev))
    # module = manifold.TSNE(n_components=2,
    #                        init='pca',
    #                        random_state=1000,
    #                        early_exaggeration=20,
    #                        method='exact')
    # x_reduced = module.fit_transform(all_vec)

    # print('REDUCED SHAPE: {}'.format(x_reduced.shape))
    # train_acc, test_acc, my_svm = svm_classifier(all_vec, all_lab, name)

    return all_vec, all_lab


def svm_classifier(vec, lab, name, class_weight_=None, fold_id=None):
    """
    Train classifier
    :param vec: input feature vectors
    :param lab: labels corresponding to feature vectors
    :param name: name of the classifier
    :param class_weight_: class weight
    :param fold_id: if using k-fold, mark the fold id
    :return: train_acc, test_acc, classifier
    """
    print('TRAINING SVM: SHAPES AS FOLLOWED!')
    print(vec.shape)
    print(lab.shape)
    gt_vectors = vec
    gt_labels = lab

    # cross_val_lab = np.reshape(lab, (-1, 1))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(gt_vectors, gt_labels, test_size=0.2,
                                                                                random_state=1)

    if class_weight_ is None:
        my_svm = sklearn.svm.SVC(kernel='poly', random_state=50, tol=1e-50, max_iter=500000, C=1, probability=True)
        cross_val_svm = sklearn.svm.LinearSVC(random_state=10, tol=1e-20, max_iter=500000, C=1)

    else:
        my_svm = sklearn.svm.LinearSVC(random_state=10, tol=1e-20, max_iter=500000, C=1, class_weight=class_weight_)
        cross_val_svm = sklearn.svm.LinearSVC(random_state=10, tol=1e-20, max_iter=500000, C=1,
                                              class_weight=class_weight_)
    print(X_train)
    print(y_train)
    print(X_train.shape, y_train.shape)
    my_svm.fit(X_train, y_train)
    with open('svm_models/{}.pickle'.format(name), 'wb') as f:
        pickle.dump(my_svm, f)
        print('{} SVM model saved!'.format(name))
    with open('svm_models/{}.pickle'.format(name), 'rb') as f:
        my_svm = pickle.load(f)
    train_result = my_svm.predict(X_train)
    test_result = my_svm.predict(X_test)
    train_acc = (y_train == train_result).sum() / float(y_train.size)
    test_acc = (y_test == test_result).sum() / float(y_test.size)

    plot_confusion_matrix(test_result, y_test, [0, 1, 2], normalize=True)
    if fold_id is None:
        plt.savefig('new_stuff/{}_svm_mx.png'.format(name))
    else:
        plt.savefig('new_stuff/{}/{}_svm_mx.png'.format(fold_id, fold_id))
    plt.show()

    print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    return train_acc, test_acc, my_svm


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def ssim_test(frames, const_loc, vid_idx):
    """
    Function for generating the actual graph, including polynomial plot, original data plot, fitted linear line,
    and gradient and quality of fit
    :param frames: container for frame images
    :param const_loc: cell location text
    :param vid_idx: video name (index, e.g. Video_XY(index))
    :return: None
    """
    cw_orig = np.zeros((1, 120))
    ccw_orig = np.zeros((1, 120))
    complx_orig = np.zeros((1, 120))

    cw_deg1 = np.zeros((1, 120))
    ccw_deg1 = np.zeros((1, 120))
    complx_deg1 = np.zeros((1, 120))

    cw_deg15 = np.zeros((1, 120))
    ccw_deg15 = np.zeros((1, 120))
    complx_deg15 = np.zeros((1, 120))

    all_ssims = []
    all_polys = []
    all_labs = []
    labs = np.loadtxt('const_index/lab/{}.txt'.format(vid_idx))
    first_frame = frames[0]
    nonrotation_color = (255, 0, 0)
    # os.makedirs('new_stuff/ssim_figs/{}'.format(vid_idx))
    for cell_idx in range(const_loc.shape[0]):
        ssim_list = []
        x, y, w, h = const_loc[cell_idx]
        _, _, cell_lab = labs[cell_idx]
        # os.makedirs('new_stuff/ssim_figs/{}/{}'.format(vid_idx, cell_idx))
        for i in range(1, len(frames)):
            frame_a = frames[0]
            patch_a = frame_a[int(y - int(h / 2)):int(y + int(h / 2) + 1), int(x - int(w / 2)):int(x + int(w / 2) + 1)]
            frame_b = frames[i]
            patch_b = frame_b[int(y - int(h / 2)):int(y + int(h / 2) + 1),
                     int(x - int(w / 2)):int(x + int(w / 2) + 1)]
            blank_a = np.zeros_like(patch_a)
            blank_a = cv2.circle(blank_a, (26, 26), 25, 255, cv2.FILLED)
            blank_a[blank_a == 255] = patch_a[blank_a == 255]
            blank_b = np.zeros_like(patch_b)
            blank_b = cv2.circle(blank_b, (26, 26), 25, 255, cv2.FILLED)
            blank_b[blank_b == 255] = patch_b[blank_b == 255]
            ssim_const = ssim(blank_a, blank_b, data_range=blank_b.max() - blank_b.min())
            # ssim_const = ssim(patch_a, patch_b, data_range=patch_b.max() - patch_b.min())
            ssim_list.append(ssim_const)
            cv2.imwrite('new_stuff/ssim_figs/{}/{}/b{}.jpg'.format(vid_idx, cell_idx, i), blank_b)
            cv2.imwrite('new_stuff/ssim_figs/{}/{}/b{}.jpg'.format(vid_idx, cell_idx, 0), blank_a)

        ori_y = np.asarray(ssim_list)
        # print('ori_y is {}'.format(ori_y))
        ori_x = np.linspace(0, ori_y.shape[0] + 1, ori_y.shape[0])
        # print('ori_x is {}'.format(ori_x))

        p = np.polyfit(ori_x, ori_y, deg=1)
        print('The linear slope of this cell is: {}'.format(p))
        print('p array shape: {}'.format(p.shape))

        p_high_degree = np.polyfit(ori_x, ori_y, deg=15)

        # convert to polynomial function
        f = np.poly1d(p)
        f_deg15 = np.poly1d(p_high_degree)

        # generate new x's and y's
        x_new = np.linspace(0, 120, ori_y.shape[0])
        x_new_deg15 = np.linspace(0, 120, ori_y.shape[0])

        y_new = f(x_new)
        y_new_deg15 = f_deg15(x_new_deg15)

        evaluation = np.mean((y_new - ori_y) ** 2)
        print('ori_y\n{}'.format(ori_y.shape))
        print('y_new\n{}'.format(y_new.shape))
        print('y_new_deg4\n{}'.format(y_new_deg15.shape))
        time.sleep(30)


        ori_y = np.reshape(ori_y, (1, ori_y.shape[0]))
        y_new = np.reshape(y_new, (1, y_new.shape[0]))
        y_new_deg15 = np.reshape(y_new_deg15, (1, y_new_deg15.shape[0]))

        # cv2.imwrite('frames/vid{}.jpg'.format(vid_idx), first_frame)
        # print('First frame of vid{} is saved'.format(vid_idx))

        print('Vid {}, cell {}/{} is {}'.format(vid_idx, cell_idx, const_loc.shape[0], cell_lab))
        print('Cell location ({}, {})'.format(x, y))
        print('cw_ori shape {}'.format(cw_orig.shape))
        print('ori_y shape {}'.format(ori_y.shape))
        print('**********')

        slope = abs(p[0])
        print('The linear slope is: {}'.format(slope))
        sub1 = np.subtract(y_new_deg15, y_new)
        sub2 = np.subtract(ori_y, y_new)
        a = np.absolute(sub1)
        b = np.absolute(sub2)
        avgdifference = np.mean(a)
        avgdifference2 = np.mean(b)
        print('The average difference is: {}'.format(avgdifference))
        print('The second average difference between red and blue is: {}'.format(avgdifference2))
        if slope < .0005 and avgdifference < 0.05:
            print('This is a nonmoving cell')
            cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(first_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
            cv2.putText(first_frame, '{}'.format(cell_idx),
                       (int(x-w/2), int(y-h/2)),
                       cv2.FONT_HERSHEY_COMPLEX,
                       1,
                       color=(255, 0, 0))
            cv2.imwrite('frames/vid{}.jpg'.format(vid_idx), first_frame)
            print('SAVED FOR CELL {}'.format(cell_idx))
            # time.sleep(30)
            # cv2.waitKey(0)
        else:
           print('This is a moving cell')


       # # *****************************************************
       # slope = abs(p[0])
       # print('The linear slope is: {}'.format(slope))
       # out_arr = np.subtract(y_new_deg15, y_new)
       # a = np.absolute(out_arr)
       # avgdifference = np.mean(a)
       #
       #
       # if slope < .0005 and avgdifference < 0.05:
       #   print('This is a nonmoving cell')
       # else:
       #   print('This is a moving cell')
       #
       #
       # Reference Info
       # print('cell number {}'.format(cell_idx))
       # time.sleep(30)
       #
       # #SSIM vales for yellow line
       # print('y_new_deg15 is {}'.format(y_new_deg15))
       # #SSIM values for red line
       # print('y_new is {}'.format(y_new))
       #
       # out_arr = np.subtract(y_new_deg15, y_new)
       # a = np.absolute(out_arr)
       # avgdifference = np.mean(a)
       #
       # diff = np.empty((1,120))
       # while(cell_idx < 121)
       #   arr[cell_idx] = abs(y_new_deg15-y_new)
       #
       # a = np.array([[1, 2], [3, 4]])
       # avgdifference = np.mean(a)
       # # *******************************************************

       if cell_lab == 0:  # CW
           cw_orig = np.concatenate((cw_orig, ori_y), axis=0)
           cw_deg1 = np.concatenate((cw_deg1, y_new), axis=0)
           cw_deg15 = np.concatenate((cw_deg15, y_new_deg15), axis=0)
       elif cell_lab == 1: # CCW
           ccw_orig = np.concatenate((ccw_orig, ori_y), axis=0)
           ccw_deg1 = np.concatenate((ccw_deg1, y_new), axis=0)
           ccw_deg15 = np.concatenate((ccw_deg15, y_new_deg15), axis=0)
       else:
           complx_orig = np.concatenate((complx_orig, ori_y), axis=0)
           complx_deg1 = np.concatenate((complx_deg1, y_new), axis=0)
           complx_deg15 = np.concatenate((complx_deg15, y_new_deg15), axis=0)
       print(cw_orig.shape, cw_deg1.shape, cw_deg15.shape)
       print(ccw_orig.shape, ccw_deg1.shape, ccw_deg15.shape)
       print(complx_orig.shape, complx_deg1.shape, complx_deg15.shape)
       print('=' * 20)
       time.sleep(.03)

       # plt.ylim(0.0, 1.0)
       # plt.plot(ori_y, color='b', label='Original')
       # plt.plot(y_new, color='red', label='1 degree fit')
       # plt.plot(y_new_deg15, color='orange', label='high degree fit')
       # plt.title('Cell {}, label {} \n {} {}'.format(cell_idx, labs[cell_idx][2], p, evaluation))
       # plt.legend(loc='lower left')
       # plt.show()
       # plt.savefig('new_stuff/ssim_figs/{}/{}/aplot{}.png'.format(vid_idx, cell_idx, cell_idx))
       # plt.clf()
       # time.sleep(30)

       '''
           0 is for CW and CCW
           1 is for other
       '''

       # all_ssims.append(ssim_list)
       # if labs[cell_idx][2] == 2:
       #     lab = 1
       # else:
       #     lab = 0
       # all_labs.append(lab)
       # tmp = [p[0], p[1], evaluation]
       # all_polys.append(tmp)

    # all_ssims = np.asarray(all_ssims)
    # all_labs = np.asarray(all_labs)
    # all_polys = np.asarray(all_polys)
    # print(all_ssims.shape, all_labs.shape)
    # # np.savetxt('new_stuff/ssims/{}_vec.txt'.format(vid_idx), all_ssims)
    # # np.savetxt('new_stuff/ssims/{}_lab.txt'.format(vid_idx), all_labs)
    # np.savetxt('new_stuff/polys/{}_vec.txt'.format(vid_idx), all_polys)
    # np.savetxt('new_stuff/polys/{}_lab.txt'.format(vid_idx), all_labs)
    # print('finished video {}'.format(vid_idx))



    np.savetxt('vectors/cw/origin/{}.txt'.format(vid_idx), cw_orig)
    np.savetxt('vectors/cw/deg1/{}.txt'.format(vid_idx), cw_deg1)
    np.savetxt('vectors/cw/deg15/{}.txt'.format(vid_idx), cw_deg15)
    np.savetxt('vectors/ccw/origin/{}.txt'.format(vid_idx), ccw_orig)
    np.savetxt('vectors/ccw/deg1/{}.txt'.format(vid_idx), ccw_deg1)
    np.savetxt('vectors/ccw/deg15/{}.txt'.format(vid_idx), ccw_deg15)
    np.savetxt('vectors/complx/origin/{}.txt'.format(vid_idx), complx_orig)
    np.savetxt('vectors/complx/deg1/{}.txt'.format(vid_idx), complx_deg1)
    np.savetxt('vectors/complx/deg15/{}.txt'.format(vid_idx), complx_deg15)
    print('all are saved for video {}'.format(vid_idx))



if __name__ == '__main__':
    # cell_types = ['cw', 'ccw', 'complx']
    # line_types = ['origin', 'deg1', 'deg15']
    # for cell_type in cell_types:
    #     for line_type in line_types:
    #         # print('cell_type {}, line_type {}'.format(cell_type, line_type))
    #         fold_path = 'vectors/{}/{}'.format(cell_type, line_type)
    #         if os.path.isdir(fold_path) == False:
    #             os.makedirs(fold_path)
    #             print('{} does not exist, made one for you'.format(fold_path))
    '''
    Load videos -> calculate the 1v120 lines/plots and gradient & quality data -> mannual identity or train a classifier
    '''

    '''LOADING VIDEO'''
    for video_id in range(15, 16):
        print()
        print()
        print('==============STARTING VIDEO {}=============='.format(video_id))
        print()
        print()
        if video_id < 10:
            video_id = '0{}'.format(video_id)

        frames = []
        const_loc = np.loadtxt("const_index/loc/{}.txt".format(video_id))
        frame_gap = 1
        ssim_set_val = 0.80
        all_pairs = []
        load_video(frames, frame_gap, video_id)
        print('Video loaded {} frames'.format(frames.__len__()))
        # matching1()
        # matching2(frames, const_loc, all_pairs)
        # get_angles(all_pairs, video_id, const_loc, frames)
        # ana_results()
        # test()
        ssim_test(frames, const_loc, video_id)
        print()
        print()
        print('==============FINISHED VIDEO {}=============='.format(video_id))
        print()
        print()
    time.sleep(30)

    '''BIAS FREE SVM TRAINING'''
    # vecs = []
    # labs = []
    # test_vec = []
    # test_lab = []
    # for video_id in range(1, 33):
    #     if video_id not in test_set_index:
    #         if video_id < 10:
    #             video_id = '0{}'.format(video_id)
    #         one_vec = np.loadtxt('new_stuff/vectors/vec_{}.txt'.format(video_id))
    #         one_lab = np.loadtxt('const_index/lab/{}.txt'.format(video_id))
    #
    #         for row in one_vec:
    #             vecs.append(row)
    #
    #         for row in one_lab:
    #             labs.append(row[2])
    #     else:
    #         if video_id < 10:
    #             video_id = '0{}'.format(video_id)
    #         one_vec = np.loadtxt('new_stuff/vectors/vec_{}.txt'.format(video_id))
    #         one_lab = np.loadtxt('const_index/lab/{}.txt'.format(video_id))
    #
    #         for row in one_vec:
    #             test_vec.append(row)
    #
    #         for row in one_lab:
    #             test_lab.append(row[2])
    #
    # vecs = np.asarray(vecs)
    # labs = np.asarray(labs)
    #
    # balanced_vec, balanced_lab = bias_free('none', vecs, labs, two_class=False)
    # np.savetxt('new_stuff/balanced_vec.txt', balanced_vec)
    # np.savetxt('new_stuff/balanced_lab.txt', balanced_lab)
    # svm_classifier(balanced_vec, balanced_lab, '120_rot_2')

    # vecs = []
    # labs = []
    # for i in range(1, 33):
    #     if i not in test_set_index:
    #         if i < 10:
    #             i = '0{}'.format(i)
    #         vec = np.loadtxt('new_stuff/polys/{}_vec.txt'.format(i))
    #         lab = np.loadtxt('new_stuff/polys/{}_lab.txt'.format(i))
    #         for x in range(vec.shape[0]):
    #             tmp = [vec[x][0], vec[x][2]]
    #             vecs.append([vec[x][0]*vec[x][2]])
    #             labs.append(lab[x])
    #
    # vecs = np.asarray(vecs)
    # labs = np.asarray(labs)
    #
    # print(vecs.shape, labs.shape)
    # train_vec, train_lab = bias_free('none', vecs, labs, two_class=True, use_flip=False)
    # np.savetxt('new_stuff/poly_train_vec.txt', train_vec)
    # np.savetxt('new_stuff/poly_train_lab.txt', train_lab)
    #
    # svm_classifier(train_vec, train_lab, 'poly_filter_oth')

    '''MANUAL CLASSIFY'''
    vecs = []
    labs = []
    for i in range(1, 33):
        if i < 10:
            i = '0{}'.format(i)
        vec = np.loadtxt('new_stuff/polys/{}_vec.txt'.format(i))
        lab = np.loadtxt('new_stuff/polys/{}_lab.txt'.format(i))
        for x in range(vec.shape[0]):
            tmp = [vec[x][0], vec[x][2]]
            # vecs.append([vec[x][0] * vec[x][2]])
            vecs.append(tmp)
            labs.append(lab[x])

    vecs = np.asarray(vecs)
    labs = np.asarray(labs)

    print(vecs.shape, labs.shape)

    preds = []
    rots_grad = []
    oth_grad = []
    rots_qual = []
    oth_qual = []
    idx = 0
    gradients = []
    qualities = []

    for row in vecs:
        gradient, quality = row
        # if gradient > - 0.0005 and quality > 0.001 or gradient > -0.0005 and quality < 0.0005:
        # if gradient < -0.0008 and quality < 0.0005:
        gradients.append([gradient])
        qualities.append([quality])
        if labs[idx] == 1:
            oth_grad.append(round(gradient, 4))
            oth_qual.append(quality)
        else:
            rots_grad.append(round(gradient, 4))
            rots_qual.append(quality)
        idx += 1

    print('OTH: {} {}'.format(mean(oth_grad), mean(oth_qual)))
    print('ROT: {} {}'.format(mean(rots_grad), mean(rots_qual)))

    oth_grad = np.asarray(oth_grad)
    rots_grad = np.asarray(rots_grad)

    hist_oth_grad = np.histogram(oth_grad)
    hist_rots_grad = np.histogram(rots_grad)

    hist_oth_qual = np.histogram(oth_qual)
    hist_rots_qual = np.histogram(rots_qual)
    print()
    print(hist_oth_grad[0])
    print(hist_oth_grad[1])
    print(hist_rots_grad[0])
    print(hist_rots_grad[1])
    print()

    print()
    print(hist_oth_qual[0])
    print(hist_oth_qual[1])
    print(hist_rots_qual[0])
    print(hist_rots_qual[1])
    print()

    # plt.hist(oth_grad)
    # plt.show()
    # plt.clf()
    # plt.hist(rots_grad)
    # plt.show()
    # plt.clf()
    #
    # plt.hist(oth_qual)
    # plt.show()
    # plt.clf()
    # plt.hist(rots_qual)
    # plt.show()
    # plt.clf()

    for row in vecs:
        gradient, quality = row
        if gradient > -0.0004 and quality < 0.0003:
            preds.append(1)
        else:
            preds.append(0)
        idx += 1

    preds = np.asarray(preds)
    print(preds)
    print(labs)

    plot_confusion_matrix(preds, labs, [0, 1], normalize=True)
    plt.show()

    gradients = np.asarray(gradients)
    qualities = np.asarray(qualities)

    # bias_free_grad_vec, bias_free_grad_lab = bias_free('none', gradients, labs, two_class=True, use_flip=False)
    # bias_free_qual_vec, bias_free_qual_lab = bias_free('none', qualities, labs, two_class=True, use_flip=False)
    #
    # svm_classifier(bias_free_grad_vec, bias_free_grad_lab, 'bias_free_grad')
    # svm_classifier(bias_free_qual_vec, bias_free_qual_lab, 'bias_free_qual')





