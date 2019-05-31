# Test the trained networks
# Created by yanrpi @ 2018-08-01
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models.resnet as resnet
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import filecmp
from os import path
from PIL import Image
import resnet_regression as resnet
import seaborn as sns
import filecmp
import copy

######################################

# datafolder = '181007_50N_A'
# datafolder = '181031_multifc2'
# datafolder = '181031_multifc'
# datafolder = '181102_1channel'
# datafolder = '181107_avgtest'
# datafolder = 'cross_val_10_new'
# datafolder = 'cross_val_10'
# datafolder = '181114_slice'
# datafolder = '181120_context'
# datafolder = '190104_multicontext'
# datafolder = '01slice'
# datafolder = '01slice_origin'
# datafolder = '01slice_unresize'
# datafolder = '02patch'
# datafolder = '02patch_origin'
datafolder = 'raw'

angles = np.loadtxt('rand_angle.txt')
roi_w = 160 // 2
roi_h = 160 // 2
roi_d = 3 // 2
lower_b = -200
upper_b = 500
training_part = 'patch'
start_fold = 0
end_fold = 10
vec_mean = [177.68223469, 139.43425626, 179.30670566]
vec_std = [41.99594637, 51.20869017, 46.1423163]
container = np.ones((1, 2))
retrain_folds = np.asarray([0])


# Just normalization for validation
data_transform = transforms.Compose([
        # transforms.Resize(int(224*1.5)),
        # transforms.Resize(320),

        # transforms.Resize(320),
        transforms.CenterCrop(52),
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(vec_mean, vec_std)
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network = ['ResNet-101', 'ResNet-152']
network = ['ResNet-34']

# outfile = open('acc_results/fc'+network[0] + '.txt', 'a+')

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

# input the weights of neurons of fc layers
# output the normalized weights
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

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


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

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

        # find last .png
        # extract the code right before it
        # this is new
        img_id = img_path[-19: -13]
        # this is old HyRiskNet
        # img_id = img_path[-16: -10]
        agatston_score = float(img_path[-9]) * 2 / 3 - 1
        manual_score = float(img_path[-5]) * 2 / 3 - 1
        # life_label = float(img_path[-9])
        idx = np.where(center_list[:, 0] == int(img_id))[0][0]
        center = center_list[idx, 1:]

        # print('img_path {}'.format(img_path))
        # print('center list {}'.format(center_list))

        #img_name = os.path.join(self.root_dir,
        #                        self.image_filenames[idx])
        #image = io.imread(img_path)
        with open(img_path, 'rb') as f:
            # print('what is f {}'.format(f))
            # time.sleep(30)
            img = Image.open(f)
            image = img.convert('RGB')

            xl = center[0] - roi_w - 1
            xu = center[0] + roi_w
            yl = center[1] - roi_h - 1
            yu = center[1] + roi_h
            patch = image.crop((xl, yl, xu, yu))


        # time.sleep(30)
        # plt.imshow(patch)
        # time.sleep(30)


        if self.transform:
            image = self.transform(image)
            patch = self.transform(patch)

        return image, patch, target, agatston_score, manual_score, img_id


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
        if angle < 0:
            target = 0
        elif angle > 0:
            target = 1
        else:
            target = 2
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
        # print('image loading shape {}'.format(image.shape))
        return image, target, img_id, angle

def save_slices(inputs, list, preds):
    x_numpy = np.asarray(inputs)
    preds_numpy = np.asarray(preds)
    for i in range(0, x_numpy.shape[0]):
        # slice = x_numpy[i, 0, :, :]
        # normalized_slice = array_normalize(slice)
        # cv2.imwrite('experiments/heatmaps/{}(L{}P{})_slice.jpg'.format(list[4][i],
        #                                                    list[3][i],
        #                                                    preds_numpy[i]), normalized_slice)

        image = x_numpy[i, :, :, :]
        image_c0 = np.reshape(image[0, :, :], (image.shape[1], image.shape[2], 1))
        image_c1 = np.reshape(image[1, :, :], (image.shape[1], image.shape[2], 1))
        image_c2 = np.reshape(image[2, :, :], (image.shape[1], image.shape[2], 1))
        image = np.concatenate((image_c2, image_c1), axis=2)
        image = np.concatenate((image, image_c0), axis=2)
        image = array_normalize(image)
        cv2.imwrite('experiments/heatmaps_{}/{}_L{}P{}_{}.jpg'.format(training_part, list[4][i], list[3][i],
                                                                   preds_numpy[i], training_part), image)

def save_heatmap(featuremap, img_id, class_id):
    # ax = sns.heatmap(featuremap, vmin=0, vmax=255)
    # ax = sns.heatmap(featuremap, robust=True)
    # fig = ax.get_figure()
    # fig.savefig('experiments/heatmaps_{}/{}_P{}_{}.jpg'.format(training_part, img_id, class_id, training_part))
    # fig.clf()
    plt.imsave('experiments/heatmaps_{}/{}_P{}_{}.jpg'.format(training_part, img_id, class_id, training_part), featuremap,
               cmap='jet_r')

def test_model(model):
    '''Test the trained models'''

    since = time.time()

    test_scores = []
    test_labels = []
    running_corrects = 0

    # file = open('data/calc_sen_spe/sen-spe{}-{}.txt'.format(net, training_part), 'a')
    # file2 = open('data/prob&label/{}_{}.txt'.format(datafolder, net), 'a')
    # file3 = open('data/idx{}_{}.txt'.format(datafolder, net), 'a')

    # Iterate over data.
    for inputs, labels, img_id, angle in dataloader:
        labels = labels.type(torch.LongTensor)
        inputs = inputs.to(device)
        labels = labels.to(device)
        angle = angle.to(device)

        # img1 = inputs.data.cpu().numpy()[0, 0, :, :]
        # print('img1 shape {}'.format(img1.shape))
        # cv2.imwrite('img1.jpg', array_normalize(img1))
        # print('img1 saved')
        # time.sleep(30)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        print('inputs shape {}'.format(inputs.shape))
        print('outputs shape {}'.format(outputs.shape))
        print('label shape {}'.format(labels.shape))
        print('preds shape {}'.format(preds.shape))
        print('preds\n{}'.format(preds))
        print('outputs\n{}'.format(outputs))
        print('label\n{}'.format(labels))
        print('angle\n{}'.format(angle))
        print(img_id)
        time.sleep(30)
        # print('label {}'.format(labels))

        ''' Save the corresponding image with label and prediction '''
        # if training_part == 'slice':
        #     save_slices(inputs, list, preds)
        # else:
        #     save_slices(input_patch, list, preds)
        # print('slices have been saved!')
        # time.sleep(30)

        # preds_np = preds.data.cpu().numpy()
        # label_np = labels.data.cpu().numpy()
        # preds_np = np.reshape(preds_np, (preds_np.shape[0], 1))
        # label_np = np.reshape(label_np, (label_np.shape[0], 1))
        # results_np = np.concatenate((label_np, preds_np), axis=1)
        # np.savetxt(file, results_np)
        #
        # probs_label = np.concatenate((network_probs, network_probs), axis=1)
        # probs_label = np.concatenate((probs_label, label_np), axis=1)
        # np.savetxt(file2, probs_label)

        # probs_label = np.concatenate((network_probs, network_probs), axis=1)
        # probs_label = np.concatenate((probs_label, label_np), axis=1)
        # img_id = img_id.data.cpu().numpy()
        img_id = np.asarray(img_id, dtype=np.float64)
        img_id = np.reshape(img_id, (img_id.shape[0], 1))
        # print('img_id shape {}'.format(img_id.shape))
        # print(img_id)
        # time.sleep(30)
        # np.savetxt(file3, img_id)

        test_scores.extend(outputs.data.cpu().numpy()[:, 1])
        test_labels.extend(labels.data.cpu().numpy())

        running_corrects += torch.sum(preds == labels.data)
    #time.sleep(10)
    # file.close()
    # file2.close()
    # file3.close()
    acc = running_corrects.double() / dataset_size

    # outfile.write('{:.4f} '.format(acc))
    print('Test acc: {:.4f}'.format(acc))
    # file = open('data/results.txt', 'a')
    # file.write(acc)
    # file.close()

    time_elapsed = time.time() - since
    #print('*'*10 + 'Test complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    #print()
    # print(test_scores)
    # print('test scores size {}'.format(len(test_scores)))
    # print(test_labels)
    # print('test labels size {}'.format(len(test_labels)))
    # print(acc)
    return test_scores, test_labels, acc


if __name__ == '__main__':
    k_tot = 10
    print(os.getcwd())
    avg_acc = 0
    #time.sleep(10)

    for net in network:
        if net == 'ResNet-18':
            base_model = resnet.resnet18
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
            #continue
        else:
            print('The network of {} is not supported!'.format(net))

        #file.write('safdsaf')
        # outfile = open('acc_results/fc' + net + '.txt', 'a+')
        for k in retrain_folds:
        # for k in range(start_fold, end_fold):
            print('what Cross validating fold {}/{} of {}'.format(k + 1, k_tot, net))
            # data_dir = path.expanduser('~/tmp/{}/fold_{}'.format(datafolder, k))
            # data_dir = path.expanduser('/zion/guoh9/kaggle/cancer/{}'.format(datafolder))
            data_dir = path.expanduser('/zion/guoh9/mic_data/data')
            # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
            #                                        data_transforms[x])
            #                for x in ['train', 'val']}
            # image_datasets = {x: MortalityRiskDataset(os.path.join(data_dir, x),
            #                                         data_transforms[x])
            #                 for x in ['train', 'val']}
            image_datasets = RotationAngle(os.path.join(data_dir, 'val'),
                                                 data_transform)

            dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=32,
                                                         shuffle=False, num_workers=0)

            dataset_size = len(image_datasets)

            # class_names = image_datasets.classes

            #entry = [net]
            #print(entry)
            #print('what is entry')
            ft_acu = 0
            conv_acu = 0
            scratch_acu = 0

            #for mode in ['ft', 'conv', 'scratch']:
            for mode in ['scratch']:
                model_x = base_model(pretrained=False)
                num_ftrs = model_x.fc.in_features

                model_x.fc = nn.Linear(num_ftrs, 3)
                # model_x.fc = nn.Linear(513, 2)

                #data_dir = path.expanduser('~/tmp/my_cross_val_10/fold_0')
                fn_best_model = os.path.join(data_dir, 'best_{}_{}.pth'.format(mode, net))
                # print(fn_best_model)
                model_x.load_state_dict(torch.load(fn_best_model))
                model_x.eval()

                model_x = model_x.to(device)

                print(mode+': ', end='')
                scores, labels, myacc = test_model(model_x)
                avg_acc = avg_acc + myacc
                #entry.append(str(myacc))


                results = np.asarray([scores, labels])
                fn_results = os.path.join(data_dir, 'test_results_{}_{}.npy'.format(mode, net))
                np.save(fn_results, results)
                # print(scores.shape)
                # print(labels.shape)
            # outfile.write('\n')
        # outfile.close()
    print('avg_acc = {:.4f}'.format(avg_acc/10))
