import urllib.request
import cv2
import numpy as np
import os
import time
from os import path
import pickle
import sklearn
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import simple_cnn as sic
# import imutils

cluster = 3
degree_range = 20
vector_dimension = 24

result_folder = path.expanduser('~/tmp/rotation_results')
source_folder = path.expanduser('~/videos')
svm_file = 'my_svm.pickle'

def creat_pos_n_neg():
    for file_type in['neg']:

        for img in os.listdir(file_type):
            if file_type == 'neg':
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)
            """
            elif file_type == 'pos':
                line = file_type + '/' + img + '1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)
            """
    print('neg txt written!')

def prepare_frames(video_name):
    source_video = path.join(source_folder, video_name)
    cap = cv2.VideoCapture(source_video)
    # cap.open(source_video)
    # cap = cv2.VideoCapture()

    video_folder = path.join(result_folder, video_name[0:-4])
    if not path.isdir(video_folder):
        os.makedirs(video_folder)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gap = math.floor(length / (vector_dimension + 1))
    print('length of the video: {} frames'.format(length))
    print('gap is {}'.format(gap))
    print('Is cap opened? {}'.format(cap.isOpened()))
    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('frame', gray)
        if count % gap == 0:
            index = int(count / gap)
            filename = path.join(video_folder, 'frame{}.jpg'.format(index))
            cv2.imwrite(filename, gray)
            print('At {}, frame{}.jpg has been saved'.format(count, index))

        if count / gap == vector_dimension:
            break
        count = count + 1

    cap.release()

def cascade_detect(img):
    # img = cv2.imread('{}.jpg'.format(img_name), 0)
    file = open(txt_path, 'w')
    # file = open('singleframe4.txt', 'w')

    print('1')
    # Here load the detector xml file, put it to your working directory
    # cell_cascade = cv2.CascadeClassifier('mydata/cascade.xml')
    cell_cascade = cv2.CascadeClassifier('cascade.xml')
    # cell_cascade = cv2.CascadeClassifier('cells-cascade-20stages.xml')
    print(cell_cascade)
    print('2')
    # Here used cell_cascade to detect cells, with size restriction the detection would be much faster
    cells = cell_cascade.detectMultiScale(img, minSize=(30, 30), maxSize=(55, 55))
    # cells = cell_cascade.detectMultiScale(img, maxSize=(50, 50))
    print('3')
    print(cells)
    # time.sleep(30)
    # Here we draw the result rectangles, (x, y) is the left-top corner coordinate of that triangle
    # We can just use (x, y) to locate each cell
    # w, h are the width and height
    i = 0
    for (x, y, w, h) in cells:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img, '{}'.format(i),
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    color=(255, 255, 255))
        center_x = x + round(w / 2)
        center_y = y + round(h / 2)
        file.write('{} {} {} {}\n'.format(center_x, center_y, w, h))
        i = i+1

    file.close()
    detection_img = path.join(video_folder, 'detection.jpg')
    cv2.imwrite(detection_img, img)
    # cv2.imwrite('detection.jpg', img)

def get_rotation_template(cell_index, frame_index, angle):
    # out_folder = 'cuts/{}'.format(index)
    # if not path.isdir(out_folder):
    #     os.makedirs(out_folder)
    frame_path = path.join(video_folder, 'frame{}.jpg'.format(frame_index))
    img = cv2.imread(frame_path, 0)

    # print('cells type shape {}'.format(cells_loc.shape))
    # time.sleep(30)
    # print('1')
    row, col = img.shape
    center = tuple([cells_loc[cell_index][0], cells_loc[cell_index][1]])
    x, y, w, h = cells_loc[cell_index]
    # print(center)

    # In OpenCV, clockwise degree is negative, counter-clockwise degree is positive
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # print(rot_mat)
    result = cv2.warpAffine(img, rot_mat, (col, row))
    cv2.rectangle(result, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
    cut = result[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    # cv2.imwrite('cuts/{}/{}_{}.jpg'.format(index, index, angle), cut)
    return cut

def single_cell_direction(cell_index, frame_index):
    img_template = get_rotation_template(cell_index, frame_index+1, 0)
    degrees = np.arange(-degree_range, degree_range+1, 1)
    score_list = []
    for degree in degrees:
        img_compare = get_rotation_template(cell_index, frame_index, degree)
        ssim_const = ssim(img_template, img_compare,
                          data_range=img_compare.max() - img_compare.min())
        # print('{}: ssim compare {:.4f}'.format(degree, ssim_const))
        score_list.append(ssim_const)

    max_value = max(score_list)
    max_index = score_list.index(max(score_list))
    best_degree = max_index - degree_range

    # print('Max value: {}'.format(max_value))
    # print('Max index: {}'.format(max_index))
    print('Cell {}/{} best degree: {}'.format(cell_index+1, cells_loc.shape[0], best_degree))

    # cv2.circle(img, (int(cells_loc[0][0]), int(cells_loc[0][1])), 5, (255, 0, 0), thickness=-1)
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)
    return best_degree

def single_frame_detection(cells_loc, frame_index):
    cells_degrees = np.zeros((cells_loc.shape[0], 1))
    for i in range(0, cells_loc.shape[0]):
        degree = single_cell_direction(i, frame_index)
        cells_degrees[i][0] = degree
    # np.savetxt('frame1_rotations.txt', cells_degrees)
    # print('frame {} detection completed\n'.format(frame_index))
    # cells_degrees = np.reshape(cells_degrees, ())
    # print('cells_degree shape {}'.format(cells_degrees.shape))
    return cells_degrees

def draw_histograms(video_degrees):
    for i in range(0, video_degrees.shape[0]):
        intensity = video_degrees[i, :]
        bins = np.arange(0, 24, 1)
        plt.figure()
        plt.plot(bins, intensity, marker='o', mec='r', mfc='w', label=u'y=x^2曲线图')
        plt.hlines(0, 24, 0.5, colors="c", linestyles="dashed")
        plt.xlim((0, 24))
        plt.ylim((-20, 20))
        plt.xlabel('Frames')
        plt.ylabel('Degree')
        plt.title('Cell {}: Rotation Degrees'.format(i))
        plt.savefig('data/01_histograms/Cell{}.jpg'.format(i))
        print('{}/{}: histogram written'.format(i+1, video_degrees.shape[0]))
    print('all histograms have been written')

def draw_kmeans(video_degrees):
    print('video_degrees shape {}'.format(video_degrees.shape))
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(video_degrees)
    result_labels = kmeans.labels_
    print('labels {}'.format(result_labels))
    print('labels shape {}'.format(result_labels.shape))

    for i in range(0, cells_loc.shape[0]):
        if result_labels[i] == 1:
            color = (255, 0, 0)
        elif result_labels[i] == 0:
            color = (0, 255, 0)
        elif result_labels[i] == 2:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.circle(img, (int(cells_loc[i][0]), int(cells_loc[i][1])), 15, color, thickness=-1)
        cv2.putText(img, '{}'.format(i),
                    (int(cells_loc[i][0])-25, int(cells_loc[i][1])-25),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    color=color)

    cv2.imwrite('video_kmeans.jpg', img)
    print('video_kmeans.jpg wirtten')
    # cv2.imshow('rotation', img)
    # cv2.waitKey(0)

    time.sleep(30)

def draw_frames(video_degrees):
    for frame in range(0, video_degrees.shape[1]):
        for i in range(0, video_degrees.shape[0]):
            if video_degrees[i][frame] > 0:
                color = (0, 255, 0)
            elif video_degrees[i][frame] < 0:
                color = (255, 0, 0)
            else:
                continue
            # frame_path = path.join()
            # img = cv2.imread()
            cv2.circle(img, (int(cells_loc[i][0]), int(cells_loc[i][1])), 15, color, thickness=-1)
            cv2.putText(img, '{}'.format(i),
                        (int(cells_loc[i][0]) - 25, int(cells_loc[i][1]) - 25),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        color=color)

        cv2.imwrite('data/01_rotations/frame{}_rotation.jpg'.format(frame), img)
        print('{}/{}: rotation.jpg written'.format(frame+1, video_degrees.shape[1]))
    print('all frames have been written')
    time.sleep(60)

    cv2.imshow('rotation', img)
    cv2.waitKey(0)

def svm_classifier():

    # filename = 'train/{}_svm.pickle'.format(class_name)
    gt_vectors = np.loadtxt('data/gt_vectors.txt')
    gt_labels = np.loadtxt('data/gt_labels.txt')
    gt_labels = np.reshape(gt_labels, (gt_labels.shape[0], 1))
    data = np.concatenate((gt_labels, gt_vectors), axis=1)
    np.random.shuffle(data)
    ratio = math.floor(data.shape[0] * 0.8)
    part_train = data[0:ratio]
    part_test = data[ratio:]
    # print('train{}, val{}'.format(part_train.shape, part_test.shape))

    train_data = part_train[:, 1:]
    train_labels = part_train[:, 0]
    test_data = part_test[:, 1:]
    test_labels = part_test[:, 0]

    # print('train data size {}, train label size {}'.format(train_data.shape, train_labels.shape))
    # print('test data size {}, test label size {}'.format(test_data.shape, test_labels.shape))

    train_data = sklearn.preprocessing.normalize(train_data, axis=1)
    my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-20, max_iter=1)
    # my_svm = sklearn.svm.SVC(gamma=0.001, C=100, random_state=50)

    # my_svm = svm.SVC(kernel='linear', C=0.0001)
    my_svm.fit(train_data, train_labels)
    # with open(filename, 'wb') as f:
    #     pickle.dump(my_svm, f)
    #     print('{} SVM model saved!'.format(class_name))
    # with open(filename, 'rb') as f:
    #     clf2 = pickle.load(f)
    train_result = my_svm.predict(train_data)
    test_result = my_svm.predict(test_data)
    # print('test_result shape {}'.format(test_result.shape))
    # print('test_labels shape {}'.format(test_labels.shape))
    # print(test_result)
    # test_score = my_svm.decision_function(test_data)
    # print('train_confidence {}'.format(test_score))
    # print('train_predicts {}'.format(test_result))
    # print(test_labels)
    # print(test_result)
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    return train_acc, test_acc, my_svm
    # return normalized_score
    # return test_score
    # time.sleep(30)

def cross_validation(n):
    train = []
    test = []
    models = []
    for i in range(0, n):
        train_acc, test_acc, my_svm = svm_classifier()
        train.append(train_acc)
        test.append(test_acc)
        models.append(my_svm)
        print('{}/{}: train acc {:.4f}, test acc {:.4f}'.format(i + 1, n, train_acc, test_acc))

    best_index = test.index(max(test))
    best_svm = models[best_index]
    avg_train = sum(train) / len(train)
    avg_test = sum(test) / len(test)
    avg_test = sum(test) / len(test)
    print('avg_train {:.4f}, avg_test {:.4f}'.format(avg_train, avg_test))
    print('Best from {}: train acc {:.4f}, test acc {:.4f}'.format(best_index, train[best_index], test[best_index]))
    with open('my_svm.pickle', 'wb') as f:
        pickle.dump(best_svm, f)
        print('Best SVM model saved!'.format())

def get_frame(img, cells_loc, train_result, index):
    # cv2.imshow('shopw', img)
    # cv2.waitKey(0)
    for i in range(0, cells_loc.shape[0]):
        w, h = int(cells_loc[i][2]), int(cells_loc[i][3])
        x, y = int(cells_loc[i][0] - w//2), int(cells_loc[i][1] - h//2)
        # print('({},{}) with size ({},{})'.format(x, y, w, h))
        if train_result[i] == 0:    # Blue CW
            color = (255, 0, 0)
        elif train_result[i] == 1:  # Green CCW
            color = (0, 255, 0)
        else:                       # Red Complex
            color = (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, '{}'.format(i),
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    color=color)
    result_path = path.join(video_folder, 'result{}.jpg'.format(index))
    cv2.imwrite(result_path, img)
    return img

def get_video(video_folder):
    length = video_degrees.shape[1]
    # print('frame length {}'.format(length))
    img_path = path.join(video_folder, 'frame{}.jpg'.format(1))
    video_path = path.join(video_folder, video_name[0:-4] + '_result.avi')
    img = cv2.imread(img_path, 1)
    height, width = img.shape[0], img.shape[1]

    size = (width, height)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter(video_path, fourcc, fps, size)
    video = cv2.VideoWriter()
    video.open(video_path, fourcc, fps, size, True)
    for index in range(1, length+1):
        img_path = path.join(video_folder, 'frame{}.jpg'.format(index))
        img = cv2.imread(img_path, 1)
        result = get_frame(img, cells_loc, train_result, index)
        video.write(result)
        print('frame {} written!'.format(index))
    video.release()

def save_diary(train_result, time_finished):
    cells_num = train_result.shape[0]
    num_cw = np.count_nonzero(train_result == 0)
    num_ccw = np.count_nonzero(train_result == 1)
    num_other = np.count_nonzero(train_result == 2)
    ratio_cw = 100 * num_cw / cells_num
    ratio_ccw = 100 * num_ccw / cells_num
    ratio_other = 100 * num_other / cells_num

    result_diary = path.join(video_folder, 'result_diary.txt')
    file = open(result_diary, 'w')
    file.write('{}\n'.format(video_name))
    file.write('Detected {} cells\n'.format(cells_num))
    file.write('{:.1f}% CW cells: {}\n'.format(ratio_cw, num_cw))
    file.write('{:.1f}% CCW cells: {}\n'.format(ratio_ccw, num_ccw))
    file.write('{:.1f}% Other cells: {}\n'.format(ratio_other, num_other))
    file.write('Finished in {:.0f}m {:.0f}s\n'.format(time_finished // 60, time_finished % 60))
    file.close()

    cell_index = np.arange(0, cells_num, 1)
    cell_index = np.reshape(cell_index, (cells_num, 1))
    result_status = np.reshape(train_result, (cells_num, 1))
    overall_result = np.concatenate((cell_index, result_status), axis=1)
    overall_txt = path.join(video_folder, 'index&result.txt')
    overall_result = overall_result.astype(np.uint8)
    np.savetxt(overall_txt, overall_result, fmt='%i')
    print('overall result saved!')

if __name__ =='__main__':
    # n = 50
    # cross_validation(n)
    # time.sleep(30)
    for i in range(10, 33):
        since = time.time()
        video_name = 'XY{}_video.avi'.format(i)
        # prepare_frames(video_name)
        # time.sleep(30)
        img_name = 'frame0.jpg'

        video_folder = path.join(result_folder, video_name[0:-4])
        txt_path = path.join(video_folder, '{}_loc.txt'.format(video_name[0:-4]))
        img_path = path.join(video_folder, img_name)
        vectors_path = path.join(video_folder, 'vectors.txt')

        # img = cv2.imread(img_path, 1)
        # cascade_detect(img)
        # print('detection for {} completed!'.format(video_name))
        cells_loc = np.loadtxt(txt_path)
        if not os.path.isfile(vectors_path):
            video_degrees = single_frame_detection(cells_loc, 1)
            print('inter frame 1/24 completed')
            for i in range(1, 24):
                frame_degrees = single_frame_detection(cells_loc, i)
                video_degrees = np.concatenate((video_degrees, frame_degrees), axis=1)
                print('inter frame {}/24 completed'.format(i+1))
            np.savetxt(vectors_path, video_degrees)
        # print('########## finished ##############')
        # time.sleep(60)
        #
        # video_degrees = np.loadtxt('video_rotations.txt')
        ##########################################################################################

        video_degrees = np.loadtxt(vectors_path)
        print('video_degrees size {}'.format(video_degrees.shape))
        clf2 = pickle.load(open(svm_file, 'rb'))
        train_result = clf2.predict(video_degrees)
        time_finished = time.time() - since
        save_diary(train_result, time_finished)
        # time.sleep(30)
        print('train result:\n{}'.format(train_result))
        print('train result shape:\n{}'.format(train_result.shape))
        get_video(video_folder)
    # draw_histograms(video_degrees)

    # draw_kmeans(video_degrees)

    # draw_frames(video_degrees)

    ##########################################################################################
    # Here I am gathering ground truth vectors and labels
    # They are stored in gt_vectors.txt and gt_labels.txt
    ##########################################################################################
    # video_name = '10x_XY01_video_8bit.avi'
    # video_folder = path.join(result_folder, video_name[0:-4])
    # vectors_path = path.join(video_folder, 'vectors.txt')
    # vectors_space = np.loadtxt(vectors_path)
    # label_space = np.loadtxt('data/GT/cell_label1.txt')
    # print('label size {}'.format(label_space.shape))
    # for i in range(2, 5):
    #     # video_name = 'data/GT/cell_label{}.txt'.format(i)
    #     # video_folder = path.join(result_folder, video_name[0:-4])
    #     # vectors_path = path.join(video_folder, 'vectors.txt')
    #     # vectors = np.loadtxt(vectors_path)
    #     # vectors_space = np.concatenate((vectors_space, vectors), axis=0)
    #     label = np.loadtxt('data/GT/cell_label{}.txt'.format(i))
    #     label_space = np.concatenate((label_space, label), axis=0)
    #
    # np.savetxt('data/gt_labels.txt', label_space)
    # print('training_vectors size {}'.format(label_space.shape))
    # time.sleep(60)
    ###




