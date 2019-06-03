import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import cv2
import torchvision
import resnet_regression as resnet
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from os import path
import random

import tensorflow as tf
#from skimage import io
from PIL import Image
from sklearn.metrics import roc_curve, auc

import copy

angles = np.loadtxt('rand_angle.txt')

use_last_pretrained = False
network_choice = 'mic_3d'
epoch_scratch = 150
training_progress = np.zeros((epoch_scratch, 4))
start_fold = 0
end_fold = 1
retrain_folds = np.asarray([6])
plt.ion()   # interactive mode
# network = ['ResNet-18', 'ResNet-34','ResNet-50', 'ResNet-101', 'ResNet-152']
network = ['ResNet-34']
# network = ['vgg']
# network = ['inception']

vec_mean = [177.68223469, 139.43425626, 179.30670566]
vec_std = [41.99594637, 51.20869017, 46.1423163]

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(52),
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224, scale=(0.6,0.8), ratio=(1.0,1.0)),
        # transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(vec_mean, vec_std)
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(52),
        transforms.Resize(224),
        # transforms.Resize(320),
        transforms.ToTensor(),
        # transforms.Normalize(vec_mean, vec_std)
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(vec_mean)
    std = np.array(vec_std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.npy']

class MortalityRiskDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        """
        classes, class_to_idx = find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx, IMG_EXTENSIONS)

        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img_path, target = self.samples[idx]
        print('img_path\n{}'.format(img_path))
        time.sleep(30)
        img_id = img_path[-44: -4]

        # 0a6d8d83e9a57cc0bf31a2749af2582a21b56cab

        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target, img_id

def filename_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    # print('dir {}'.format(dir))
    for filename in os.listdir(dir):
        # print(filename)
        file_path = path.join(dir, filename)
        images.append(file_path)
        # print(file_path)
    # print(images)
    return images

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

class RotationAngle(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        """
        print('root_dir {}'.format(root_dir))
        samples = filename_list(root_dir)
        # print(samples)
        # time.sleep(30)

        # classes, class_to_idx = find_classes(root_dir)
        # samples = make_dataset(root_dir, class_to_idx, IMG_EXTENSIONS)

        # self.root_dir = root_dir
        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img_path = self.samples[idx]
        img_id = img_path[-8: -4]

        '''random angle every time'''
        # target = random.randint(-5, 5)

        '''loading the same random angle'''
        index = int(img_id)
        angle = angles[index]
        # print('angles\n{}'.format(angles))
        # time.sleep(30)
        if angle < 0:
            target = 0  # CW
        elif angle > 0:
            target = 1  # CCW
        else:
            target = 2  # NR
        # angle = angle + 5

        # print('img_id {}, index {}'.format(img_id, index))
        # time.sleep(30)


        img = np.loadtxt(img_path)
        rotate_img = rotateImage(img, angle)

        # img_crop = img[26:78, 26:78]
        # img_crop = np.reshape(img_crop, (1, img_crop.shape[0], img_crop.shape[1]))
        # rot_crop = rotate_img[26:78, 26:78]
        # rot_crop = np.reshape(rot_crop, (1, rot_crop.shape[0], rot_crop.shape[1]))

        # cv2.imwrite('img_tensor2.jpg', rotate_img)
        # print('saved')
        # time.sleep(30)

        img = Image.fromarray(np.uint8(img * 255), 'L')
        rotate_img = Image.fromarray(np.uint8(rotate_img * 255), 'L')

        # rot_crop = Image.fromarray(rotate_img.astype('uint8'), 'RGB')

        # image = np.concatenate((img_crop, rot_crop), axis=0)

        # print('img_crop shape {}, rot_crop shape {}'.format(img_crop.shape, rot_crop.shape))
        # print('image shape {}'.format(image.shape))
        # print('img_id {}'.format(img_id))
        # cv2.imwrite('{}.jpg'.format(idx), img)
        # cv2.imwrite('{}_rotate.jpg'.format(idx), rotate_img)
        # time.sleep(30)

        # im = Image.fromarray(np.uint8(cm.gist_earth(myarray) * 255))

        if self.transform:
            img_crop = self.transform(img)
            rot_crop = self.transform(rotate_img)

        # img_npy = img_crop.numpy()
        # img_npy = img_npy[0, :, :]
        # img_npy = array_normalize(img_npy)
        # print(img_npy.shape)
        #
        # rot_npy = rot_crop.numpy()
        # rot_npy = rot_npy[0, :, :]
        # rot_npy = array_normalize(rot_npy)
        # print(img_npy.shape)
        #
        # cv2.imwrite('img_tensor.jpg', img_npy)
        # cv2.imwrite('img_tensor_rot.jpg', rot_npy)
        # print('saved')
        # time.sleep(30)

        image = torch.cat((img_crop, rot_crop), 0)
        # print('image shape {}'.format(image.shape))
        # print('target {}'.format(target))
        # time.sleep(30)

        return image, target, img_id, angle


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_ep = 0
    best_auc = 0.0
    lowest_loss = 100
    test_scores = []
    test_labels = []
    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, img_id, angles in dataloaders[phase]:
                # Get images from inputs
                #print('*'*10 + ' printing inputs and labels ' + '*'*10)
                labels = labels.type(torch.LongTensor)
                angles = angles.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)
                angles = angles.to(device)

                # labels = labels.view(labels.shape[0], 1)
                # print('labels {}'.format(labels))
                # print('inputs shape {}'.format(inputs.shape))

                # print(img_id)
                # print(manual_scores)
                # print(agatston_scores)
                # print(labels)
                # time.sleep(10)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    preds_angle = preds - 5

                    preds_class = (preds_angle < 0) * 0 + \
                                  (preds_angle > 0) * 1 + \
                                  (preds_angle == 0) * 2
                    preds_class = preds_class.type(torch.FloatTensor)
                    preds_class = preds_class.to(device)

                    # print('labels\n{}'.format(labels))
                    # print('preds_class\n{}'.format(preds_class))
                    # print('angles\n{}'.format(angles))
                    # print('preds_angle\n{}'.format(preds_angle))
                    # print('outputs\n{}'.format(outputs))
                    # time.sleep(30)

                    degree_loss = criterion(outputs, labels)
                    # class_loss = criterion2(preds_class, labels)
                    loss = degree_loss

                    # labels = labels.type(torch.LongTensor)
                    # labels = labels.to(device)

                    # print('loss {}'.format(loss))
                    # time.sleep(30)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # else:
                    #     scores = nn.functional.softmax(outputs, dim=1)
                    #     test_scores.extend(scores.data.cpu().numpy()[:, 1])
                    #     test_labels.extend(labels.data.cpu().numpy())


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # tv_hist[phase].append([epoch_loss])
            tv_hist[phase].append([epoch_loss, epoch_acc])

            # ''' Calculate this round of AUC score '''
            # epoch_auc = 0.0
            # if phase == 'val' and len(test_scores) != 0:
            #     # print('test_labels {}, test_scores {}'.format(test_labels.shape, test_scores.shape))
            #     fpr, tpr, _ = roc_curve(test_labels, test_scores)
            #     epoch_auc = auc(fpr, tpr)
            #     if epoch_auc < 0.5:
            #         test_scores = np.asarray(test_scores)
            #         test_scores = np.ones_like(test_scores) - test_scores
            #         test_scores = test_scores.tolist()
            #         # print('test_labels {}, test_scores {}'.format(test_labels.shape, test_scores.shape))
            #         # time.sleep(30)
            #         fpr, tpr, _ = roc_curve(test_labels, test_scores)
            #         epoch_auc = auc(fpr, tpr)
            #     print('roc_auc {:.4f}'.format(epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
            # if phase == 'val' and epoch_auc >= best_auc:
            # if phase == 'val' and epoch_loss <= lowest_loss:
                # best_auc = epoch_auc
                best_acc = epoch_acc
                lowest_loss = epoch_loss
                best_ep = epoch
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), fn_save)
                # print('**** best model updated with loss={:.4f} ****'.format(lowest_loss))
                print('**** best model updated with acc={:.4f} ****'.format(epoch_acc))


        # print('ep {}/{} - Train loss: {:.4f}, Val loss: {:.4f}'.format(
        #     epoch + 1, num_epochs,
        #     tv_hist['train'][-1][0],
        #     tv_hist['val'][-1][0]))
        print('ep {}/{} - Train loss: {:.4f} acc: {:.4f}, Val loss: {:.4f} acc: {:.4f}'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0], tv_hist['train'][-1][1],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1]))
        training_progress[epoch][0] = tv_hist['train'][-1][0]
        training_progress[epoch][1] = tv_hist['train'][-1][1]
        training_progress[epoch][1] = tv_hist['val'][-1][0]
        training_progress[epoch][3] = tv_hist['val'][-1][1]
        # training_progress[epoch][4] = epoch_auc
        np.savetxt('training_progress.txt', training_progress)


        #print('-' * 10)

        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #    phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('*'*10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*'*10 + 'Best val Acc: {:4f} at epoch {}'.format(best_acc, best_ep))
    print()

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #return model
    return tv_hist


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#
# %% 10-fold cross validation
#

k_tot = 10

for net in network:
    epoch_ft = 100
    epoch_conv = 100
    # epoch_scratch = 150
    if net == 'ResNet-18':
        base_model = resnet.resnet18
        #continue
    elif net == 'ResNet-34':
        base_model = resnet.resnet34
        #continue
    elif net == 'ResNet-50':
        base_model = resnet.resnet50
        #continue
    elif net == 'ResNet-101':
        base_model = resnet.resnet101
        #continue
    elif net == 'ResNet-152':
        base_model = resnet.resnet152
    else:
        print('The network of {} is not supported!'.format(net))

    # for k in range(2, k_tot):
    # for k in retrain_folds:
    for k in range(start_fold, end_fold):
        print('Cross validating fold {}/{} of {}'.format(k+1, k_tot, net))
        # data_dir = path.expanduser('~/tmp/{}/fold_{}'.format(datafolder, k))
        # data_dir = path.expanduser('/zion/guoh9/kaggle/cancer/{}'.format(datafolder))
        data_dir = path.expanduser('/zion/guoh9/mic_data/data')
        #image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
        #                                        data_transforms[x])
        #                for x in ['train', 'val']}
        # image_datasets = {x: MortalityRiskDataset(os.path.join(data_dir, x),
        #                                         data_transforms[x])
        #                 for x in ['train', 'val']}
        image_datasets = {x: RotationAngle(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        print('image_datasets\n{}'.format(image_datasets))
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                    shuffle=True, num_workers=0)
                    for x in ['train', 'val']}
        print('dataloaders\n{}'.format(dataloaders))
        print('size of dataloader: {}'.format(dataloaders.__sizeof__()))
        # time.sleep(30)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        # class_names = image_datasets['train'].classes

        if network_choice == 'resnet':
            model_ft = base_model(pretrained=False)
            # model_ft = base_model(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 3)
        else:
            model_ft = resnet.MIC_3D()

        model_ft.cuda()
        model_ft = model_ft.to(device)

        if use_last_pretrained:
            fn_best_model = os.path.join(data_dir, 'best_scratch_{}.pth'.format(net))
            model_ft.load_state_dict(torch.load(fn_best_model))
            model_ft.eval()
            model_ft.cuda()

        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        # criterion = nn.L1Loss()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-6)
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)

        # Train and evaluate
        fn_best_model = os.path.join(data_dir, '3d_best_scratch_{}.pth'.format(net))
        hist_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                            fn_best_model, num_epochs=epoch_scratch)
        fn_hist = os.path.join(data_dir, 'hist_scratch_{}.npy'.format(net))
        np.save(fn_hist, hist_ft)
        txt_path = path.join(data_dir, 'training_progress_{}.txt'.format(net))
        np.savetxt(txt_path, training_progress)
        print('h#' * 30)
        # break
        ######################################################################
