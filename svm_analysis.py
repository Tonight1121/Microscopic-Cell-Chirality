import numpy as np
import math
from sklearn.svm import SVC
import os

from os import path
from sklearn.model_selection import train_test_split
import sklearn
import pickle
import time
import matplotlib.pyplot as plt
from scipy import ndimage
import copy
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

outfolder = path.expanduser('~/tmp/05harp')

# ============================================================ #
jbhi_folder = path.expanduser('/zion/common/shared/JBHI')

data_folder = 'data'
patients_id = 'all_ids.txt'
# clinical_parameters = 'clinical_parameters.txt'
clinical_data = 'clinical_data.txt'
svm_epoch = 50

def load_clinical_data():
    data_path = path.join(data_folder, clinical_data)
    with open(data_path, 'r') as myfile:
        data = myfile.read().replace('\n', ' ')
    data = data.split()
    print('data {}'.format(data))
    # list convert to array
    data_numpy = np.asarray(data)
    # str convert to float
    data_numpy = data_numpy.astype(np.float)
    # reshape to 180 * 7, id, life, cac risk, emphysema, muscle, fat, Agatston
    data_numpy = np.reshape(data_numpy, (int(data_numpy.shape[0]/7), 7))
    print('data numpy \n{}'.format(data_numpy))
    result_numpy = 'clinical_numpy.txt'
    np.savetxt(path.join(data_folder, result_numpy), data_numpy)
    print('{} saved'.format(result_numpy))
    return data_numpy

def svm_loo():
    result_numpy = 'clinical_numpy.txt'
    clinical_parameters = np.loadtxt(path.join(data_folder, result_numpy))

    label = clinical_parameters[:, 1]
    data = clinical_parameters[:, 2:-1]
    loo = sklearn.model_selection.LeaveOneOut()
    loo.get_n_splits(data)

    i = 1
    for train_index, test_index in loo.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        data_train, data_test = data[train_index], data[test_index]
        label_train, label_test = label[train_index], label[test_index]
        # print(X_train, X_test, y_train, y_test)
        # data_train = sklearn.preprocessing.normalize(data_train, axis=1)
        # data_test = sklearn.preprocessing.normalize(data_test, axis=1)
        my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=100, random_state=50)
        my_svm.fit(data_train, label_train)
        train_result = my_svm.predict(data_train)
        test_result = my_svm.predict(data_test)
        test_score = my_svm.decision_function(data_test)
        r2 = sklearn.metrics.r2_score(label_test, test_score)
        # r2 = sklearn.metrics.r2_score(label_test, test_result)
        train_acc = (label_train == train_result).sum() / float(label_train.size)
        test_acc = (label_test == test_result).sum() / float(label_test.size)
        avg_acc = train_acc*0.7 + test_acc*0.3
        print('{}/{}: train acc {:.4f}, test acc {:.4f}, avg acc {:.4f}, avg_r2 {:.4f}'
              .format(i, clinical_parameters.shape[0], train_acc, test_acc, avg_acc, r2))
        i = i + 1
        # time.sleep(30)

def svm_classifier_all():
    result_numpy = 'clinical_numpy.txt'
    clinical_parameters = np.loadtxt(path.join(data_folder, result_numpy))
    # print('clinical_parameter shape {}'.format(clinical_parameters.shape))
    # time.sleep(30)
    # print('clinical_parameters {}'.format(clinical_parameters.shape))
    # print(100158 == clinical_parameters[0, 0])

    label = clinical_parameters[:, 1]
    data = clinical_parameters[:, 2:-1]
    # print('label shape {}'.format(label.shape))
    # print('data shape {}'.format(data.shape))
    data_train, data_test, label_train, label_test = train_test_split(data, label,
                                                                      test_size=0.1)
    # print('label_train&test shape {}, {}'.format(label_train.shape, label_test.shape))
    # print('data_train&test shape {}, {}'.format(data_train.shape, data_test.shape))
    # print('label_test {}'.format(label_test))
    # data_train = sklearn.preprocessing.normalize(data_train, axis=1)
    # data_test = sklearn.preprocessing.normalize(data_test, axis=1)
    # my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-10, max_iter=1)
    my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=100, random_state=50)
    my_svm.fit(data_train, label_train)
    train_result = my_svm.predict(data_train)
    test_result = my_svm.predict(data_test)
    overall_result = my_svm.predict(data)
    # print(label_test)
    # print(test_result)
    # print('R2 score: {}'.format(r2))
    train_score = my_svm.decision_function(data_train)
    test_score = my_svm.decision_function(data_test)
    test_prob = my_svm.predict_proba(data_test)
    # print(test_score)
    # print(test_result)
    # print(test_prob)
    # time.sleep(30)
    # r2 = sklearn.metrics.r2_score(label, overall_result)
    r2 = sklearn.metrics.r2_score(label_test, test_result)
    train_acc = (label_train == train_result).sum() / float(label_train.size)
    test_acc = (label_test == test_result).sum() / float(label_test.size)
    # print('train acc {:.3f}, test acc {:.3f}'.format(train_acc, test_acc))
    # time.sleep(30)
    return train_acc, test_acc, my_svm, r2

def cross_validation(n):
    train = []
    test = []
    avg = []
    models = []
    r2s =[]
    for i in range(0, n):
        train_acc, test_acc, my_svm, r2 = svm_classifier_all()
        # train_acc, test_acc, my_svm = svm_classifier()
        avg_acc = train_acc*0.7 + test_acc*0.3
        train.append(train_acc)
        test.append(test_acc)
        avg.append(avg_acc)
        models.append(my_svm)
        r2s.append(r2)
        print('{}/{}: train acc {:.4f}, test acc {:.4f}, avg acc {:.4f}, avg_r2 {:.4f}'.format(i + 1, n, train_acc, test_acc, avg_acc, r2))
    best_index = r2s.index(max(r2s))
    # best_index = test.index(max(test))
    best_svm = models[best_index]
    avg_train = sum(train) / len(train)
    avg_test = sum(test) / len(test)
    avg_avg = sum(avg) / len(avg)
    avg_r2 = sum(r2s) / len(r2s)
    print('avg_train {:.4f}, avg_test {:.4f}, avg_avg {:.4f}, avg_r2 {:.4f}'.format(avg_train, avg_test, avg_avg, avg_r2))
    print('Best from {}: train acc {:.4f}, test acc {:.4f}, avg acc {:.4f}, r2 {:.4f}'.format(best_index, train[best_index], test[best_index], avg[best_index], r2s[best_index]))
    with open('data/my_svm.pickle', 'wb') as f:
        pickle.dump(best_svm, f)
        print('Best SVM model saved!'.format())

def save_figure():
    training_progress = np.loadtxt('multi_context_training_progress.txt')
    print('training progress shape {}'.format(training_progress.shape))
    plt.figure()
    plt.ylim((0, 1))
    plt.xlim((0, training_progress.shape[0]))
    plt.plot(training_progress[:, 1], color='r', label=r'Training accuracy',
                     lw=2, alpha=.8)
    plt.plot(training_progress[:, 3], color='g', label=r'Validation accuracy',
                     lw=2, alpha=.8)
    plt.hlines(0.5, training_progress.shape[0], 0.5, colors="c", linestyles="dashed",
               label=r'Random Guess')
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    # plt.title('Training Progress of Patch Stream')
    plt.title('Training Progress of Combined Streams')
    plt.legend(loc="lower right")
    plt.savefig('fcn_training.jpg')
    # plt.show()
    print('figure saved')
    time.sleep(60)

def save_probs():
    clinical_parameters = np.loadtxt('data/clinical_numpy.txt')
    best_svm = pickle.load(open('data/my_svm.pickle', 'rb'))
    label = clinical_parameters[:, 1]
    data = clinical_parameters[:, 2:-1]
    prediction = best_svm.predict(data)
    scores = best_svm.decision_function(data)
    probs = best_svm.predict_proba(data)
    accuracy = (label == prediction).sum() / float(label.size)
    print(prediction)
    print(scores)
    print(probs)
    print('overall accuracy = {:.5f}'.format(accuracy))
    ids = np.reshape(clinical_parameters[:, 0], (clinical_parameters.shape[0], 1))
    id_probs = np.concatenate((ids, probs), axis=1)
    print(id_probs)
    print(id_probs.shape)
    with open('data/id_probs.txt', 'wb') as f:
        np.savetxt(f, id_probs)
    print('id_probs saved!')
    # print(data)

def load_image(dir_name, should_list_images = False):
    img_list = os.listdir(dir_name)
    img_list = [name for name in img_list if 'png' in name.lower()]
    img_list = np.sort(img_list)
    if should_list_images:
        print("Here are the images in %s\n" % dir_name)
        print(img_list)
    return img_list

def svm_folds(fold_idx):
    result_numpy = 'clinical_numpy.txt'
    clinical_parameters = np.loadtxt(path.join(data_folder, result_numpy))

    # fold_idx = 0

    fold_train_survive = path.join(outfolder, 'fold_{}/train/survived'.format(fold_idx))
    fold_train_die = path.join(outfolder, 'fold_{}/train/died'.format(fold_idx))
    train_survive_list = load_image(fold_train_survive)
    train_die_list = load_image(fold_train_die)

    train_idx_list = []

    for i in range(0, len(train_survive_list)):
        img_name = train_survive_list[i]
        train_idx_list.append(img_name[0:6])

    for i in range(0, len(train_die_list)):
        img_name = train_die_list[i]
        train_idx_list.append(img_name[0:6])

    train_idx_list = list(set(train_idx_list))
    train_idx_array = np.asarray(train_idx_list).astype(int)
    set_ids_train = set(list(train_idx_array))
    train_idx_array = np.reshape(train_idx_array, (train_idx_array.shape[0], 1))
    train_idx_array = np.sort(train_idx_array, axis=0)

    ids_all = clinical_parameters[:, 0]
    set_ids_all = set(list(ids_all))
    set_ids_val = set_ids_all - set_ids_train
    # print('set ids all length {}'.format(len(set_ids_all)))
    # print('set ids train length {}'.format(len(set_ids_train)))
    # print('set ids val length {}'.format(len(set_ids_val)))
    print('all {}, train {}, val {}'.format(len(set_ids_all), len(set_ids_train), len(set_ids_val)))
    # time.sleep(30)

    val_idx_array = list(set_ids_val)
    val_idx_array = np.asarray(val_idx_array).astype(int)
    val_idx_array = np.reshape(val_idx_array, (val_idx_array.shape[0], 1))

    index_train = np.nonzero(train_idx_array == clinical_parameters[:, 0])[1]
    index_val = np.nonzero(val_idx_array == clinical_parameters[:, 0])[1]
    train_part = np.take(clinical_parameters, index_train, axis=0)
    val_part = np.take(clinical_parameters, index_val, axis=0)

    # print('train_part.shape {}'.format(train_part.shape))
    # print('val_part.shape {}'.format(val_part.shape))

    data_train = train_part[:, 2:-1]
    label_train = train_part[:, 1]
    data_val = val_part[:, 2:-1]
    label_val = val_part[:, 1]

    ''' Training the SVM according to each fold, then apply the model to this fold on all data '''
    my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=100, random_state=50)
    my_svm.fit(data_train, label_train)
    label = clinical_parameters[:, 1]
    data = clinical_parameters[:, 2:-1]
    prediction = my_svm.predict(data)
    prediction_train = my_svm.predict(data_train)
    prediction_val = my_svm.predict(data_val)
    scores = my_svm.decision_function(data)
    print('scores {}'.format(scores))
    time.sleep(30)

    # # Tend to test the ROC curve of SVM model
    # scores_save = np.reshape(scores, (1, scores.shape[0]))
    # labels_save = np.reshape(label, (1, label.shape[0]))
    # print('scores_save shape {}'.format(scores_save.shape))
    # print('labels_save shape {}'.format(labels_save.shape))
    # results = np.concatenate((scores_save, labels_save))
    # fn_results = path.expanduser('~/tmp/04svm/fold_{}/test_results_scratch_ResNet-50.npy'.format(fold_idx))
    # np.save(fn_results, results)

    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores)) # scores array normalize to [0,1]
    scores = np.reshape(scores, (scores.shape[0], 1))
    # print('scores {}'.format(scores))
    # print('prediction {}'.format(prediction))
    # time.sleep(30)
    probs = my_svm.predict_proba(data)
    accuracy_all = (label == prediction).sum() / float(label.size)
    accuracy_train = (label_train == prediction_train).sum() / float(label_train.size)
    accuracy_val = (label_val == prediction_val).sum() / float(label_val.size)
    print('all {:.4f}, train {:.4f}, val {:.4f}'.format(accuracy_all, accuracy_train, accuracy_val))
    svm_performance[fold_idx, :] = [accuracy_all, accuracy_train, accuracy_val]
    ids = np.reshape(clinical_parameters[:, 0], (clinical_parameters.shape[0], 1))
    id_probs = np.concatenate((ids, probs), axis=1)
    id_probs = np.concatenate((id_probs, scores), axis=1)
    label = np.reshape(label, (label.shape[0], 1))
    id_probs = np.concatenate((id_probs, label), axis=1)


    with open('data/probs_score_label/id_probs{}.txt'.format(fold_idx), 'wb') as f:
        np.savetxt(f, id_probs)
    print('id_probs saved!')

def fixed_10fold():
    aucs = np.zeros((10))
    aucs = []
    for i in range(10):
        train_path = '10fold/new_10fold/fold_{}/train_set.txt'.format(i)
        test_path = '10fold/new_10fold/fold_{}/val_set.txt'.format(i)
        train_data = np.loadtxt(train_path)
        test_data = np.loadtxt(test_path)
        # print('train_data shape {}'.format(train_data.shape))
        # print('test_data shape {}'.format(test_data.shape))
        # time.sleep(30)

        data_train = train_data[:, 2:6]
        label_train = train_data[:, 1]
        data_test = test_data[:, 2:6]
        label_test = test_data[:, 1]

        ''' Training the SVM according to each fold, then apply the model to this fold on all data '''
        my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=100, random_state=50)
        my_svm.fit(data_train, label_train)

        prediction_test = my_svm.predict(data_test)
        scores = my_svm.decision_function(data_test)
        accuracy_test = (label_test == prediction_test).sum() / float(label_test.size)

        fpr, tpr, _ = roc_curve(label_test, scores)
        auc_score = auc(fpr, tpr)

        # print('scores {}'.format(scores))
        # print('pred\n{}'.format(prediction_test))
        print('ACC {:.4f}, AUC {:.4f}'.format(accuracy_test, auc_score))
        aucs.append(auc_score)
        # aucs[i] = auc_score
        # time.sleep(30)
    mean_auc = np.mean(np.asarray(aucs))
    print('mean_auc {:.4f}'.format(mean_auc))
    time.sleep(30)

def fixed_10fold_save():
    aucs = np.zeros((10))
    aucs = []
    for i in range(10):
        train_path = '10fold/new_10fold/fold_{}/train_set.txt'.format(i)
        test_path = '10fold/new_10fold/fold_{}/val_set.txt'.format(i)
        train_data = np.loadtxt(train_path)
        test_data = np.loadtxt(test_path)
        # print('train_data shape {}'.format(train_data.shape))
        # print('test_data shape {}'.format(test_data.shape))
        # time.sleep(30)

        data_train = train_data[:, 2:6]
        label_train = train_data[:, 1]
        data_test = test_data[:, 2:6]
        label_test = test_data[:, 1]

        ''' Training the SVM according to each fold, then apply the model to this fold on all data '''
        my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=10, random_state=50)
        my_svm.fit(data_train, label_train)

        prediction_test = my_svm.predict(data_test)
        scores = my_svm.decision_function(data_test)
        accuracy_test = (label_test == prediction_test).sum() / float(label_test.size)
        test_prob = my_svm.predict_proba(data_test)

        fpr, tpr, _ = roc_curve(label_test, scores)
        auc_score = auc(fpr, tpr)

        # print('scores {}'.format(scores))
        # print('pred\n{}'.format(prediction_test))
        print('ACC {:.4f}, AUC {:.4f}'.format(accuracy_test, auc_score))
        aucs.append(auc_score)
        # aucs[i] = auc_score
        # time.sleep(30)

        img_id = data_test[:, 1]
        probs = test_prob[:, 1]
        # print('img_id {}, probs {}, label {}'.format(img_id.shape, probs.shape, label_test.shape))

        img_id = np.reshape(img_id, (img_id.shape[0], 1))
        probs = np.reshape(probs, (probs.shape[0], 1))
        label_test = np.reshape(label_test, (label_test.shape[0], 1))
        val_data = np.concatenate((img_id, probs, label_test), axis=1)
        val_data = np.repeat(val_data, 3, axis=0)
        val_data_path = path.join(jbhi_folder, 'SVM/fold_{}/'.format(i))
        if os.path.isdir(val_data_path) == False:
            os.makedirs(val_data_path)
        val_data_path = path.join(val_data_path, 'val_data.txt')
        np.savetxt(val_data_path, val_data)
        print('val_data shape {}'.format(val_data.shape))

        # time.sleep(30)

    mean_auc = np.mean(np.asarray(aucs))
    print('mean_auc {:.4f}'.format(mean_auc))
    time.sleep(30)

def kamp_10fold_save(ratio=0.38):
    dsn_folder = path.expanduser('/zion/common/shared/JBHI/color-DSN')
    # ratios = np.linspace(0, 1, 21, endpoint=True)
    svm_ratio = ratio
    aucs = []
    big_aucs = []
    for i in range(10):
        train_path = '10fold/new_10fold/fold_{}/train_set.txt'.format(i)
        test_path = '10fold/new_10fold/fold_{}/val_set.txt'.format(i)
        network_data_path = path.join(dsn_folder, 'fold_{}/val_data.txt'.format(i))
        train_data = np.loadtxt(train_path)
        test_data = np.loadtxt(test_path)
        network_data = np.loadtxt(network_data_path)

        network_data = network_data[network_data[:, 0].argsort()]

        data_train = train_data[:, 2:6]
        label_train = train_data[:, 1]
        data_test = test_data[:, 2:6]
        label_test = test_data[:, 1]
        test_ids = test_data[:, 0]
        test_ids = np.reshape(test_ids, (test_ids.shape[0], 1))

        ''' Training the SVM according to each fold, then apply the model to this fold on all data '''
        my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=10, random_state=50)
        my_svm.fit(data_train, label_train)

        prediction_test = my_svm.predict(data_test)
        scores = my_svm.decision_function(data_test)
        accuracy_test = (label_test == prediction_test).sum() / float(label_test.size)
        test_prob = my_svm.predict_proba(data_test)

        # print('test_prob shape {}'.format(test_prob.shape))
        # print('test_ids shape {}'.format(test_ids.shape))
        svm_id_probs = np.concatenate((test_ids, test_prob), axis=1)
        svm_id_probs = np.repeat(svm_id_probs, 3, axis=0)
        svm_id_probs = svm_id_probs[svm_id_probs[:, 0].argsort()]
        svm_id_probs = svm_id_probs[:, [0, 2]]

        net_id_probs = network_data[:, [0, 1]]

        labels = network_data[:, 2]
        labels = np.reshape(labels, (labels.shape[0], 1))

        combined_id_prob_label = np.concatenate((svm_id_probs, net_id_probs, labels), axis=1)
        # combined_id_prob_label = mean_roc.vote_val_data(combined_id_prob_label)

        combined_prob = svm_ratio * combined_id_prob_label[:, 1] + \
                        (1 - svm_ratio) * combined_id_prob_label[:, 3]

        img_id = np.reshape(combined_id_prob_label[:, 0], (combined_id_prob_label.shape[0], 1))
        probs = np.reshape(combined_prob, (combined_prob.shape[0], 1))
        label_test = np.reshape(combined_id_prob_label[:, 4], (combined_id_prob_label.shape[0], 1))

        val_data = np.concatenate((img_id, probs, label_test), axis=1)
        val_data_path = path.join(jbhi_folder, 'KAMP/fold_{}/'.format(i))
        if os.path.isdir(val_data_path) == False:
            os.makedirs(val_data_path)
        val_data_path = path.join(val_data_path, 'val_data.txt')
        np.savetxt(val_data_path, val_data)
        print('val_data shape {}'.format(val_data.shape))


def kamp_10fold(vote=True):
    dsn_folder = path.expanduser('/zion/common/shared/JBHI/color-DSN')
    ratios = np.linspace(0, 1, 21, endpoint=True)

    aucs = []
    big_aucs = []
    for i in range(10):
        train_path = '10fold/new_10fold/fold_{}/train_set.txt'.format(i)
        test_path = '10fold/new_10fold/fold_{}/val_set.txt'.format(i)
        network_data_path = path.join(dsn_folder, 'fold_{}/val_data.txt'.format(i))
        train_data = np.loadtxt(train_path)
        test_data = np.loadtxt(test_path)
        network_data = np.loadtxt(network_data_path)

        network_data = network_data[network_data[:, 0].argsort()]

        data_train = train_data[:, 2:6]
        label_train = train_data[:, 1]
        data_test = test_data[:, 2:6]
        label_test = test_data[:, 1]
        test_ids = test_data[:, 0]
        test_ids = np.reshape(test_ids, (test_ids.shape[0], 1))

        ''' Training the SVM according to each fold, then apply the model to this fold on all data '''
        my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=10, random_state=50)
        my_svm.fit(data_train, label_train)

        prediction_test = my_svm.predict(data_test)
        scores = my_svm.decision_function(data_test)
        accuracy_test = (label_test == prediction_test).sum() / float(label_test.size)
        test_prob = my_svm.predict_proba(data_test)

        # print('test_prob shape {}'.format(test_prob.shape))
        # print('test_ids shape {}'.format(test_ids.shape))
        svm_id_probs = np.concatenate((test_ids, test_prob), axis=1)
        svm_id_probs = np.repeat(svm_id_probs, 3, axis=0)
        svm_id_probs = svm_id_probs[svm_id_probs[:, 0].argsort()]
        svm_id_probs = svm_id_probs[:, [0, 2]]

        net_id_probs = network_data[:, [0, 1]]

        labels = network_data[:, 2]
        labels = np.reshape(labels, (labels.shape[0], 1))

        combined_id_prob_label = np.concatenate((svm_id_probs, net_id_probs, labels), axis=1)
        if vote:
            combined_id_prob_label = mean_roc.vote_val_data(combined_id_prob_label)

        combined_aucs = []
        for svm_ratio in ratios:
            combined_prob = svm_ratio * combined_id_prob_label[:, 1] + \
                            (1 - svm_ratio) * combined_id_prob_label[:, 3]
            this_fpr, this_tpr, _ = roc_curve(combined_id_prob_label[:, 4].astype(int), combined_prob)
            this_auc = auc(this_fpr, this_tpr)
            combined_aucs.append(this_auc)
            # print(combined_prob.shape)
            # time.sleep(30)
        combined_aucs = np.asarray(combined_aucs)
        big_aucs.append(combined_aucs)
        # print('ratios\n{}'.format(ratios))
        # print('combined_aucs\n{}'.format(combined_aucs))

        plt.figure()
        plt.title('KAMP Fold_{}'.format(i))
        plt.plot(ratios, combined_aucs)
        plt.xlabel('SVM Ratio')
        plt.ylabel('AUC Score')
        plt.xlim((0, 1))
        # plt.ylim((0.6, 1))
        plt.savefig('plots/kamp_ratios/fold_{}.jpg'.format(i))

        fpr, tpr, _ = roc_curve(labels, svm_id_probs[:, 1])
        svm_auc = auc(fpr, tpr)

        fpr, tpr, _ = roc_curve(labels, net_id_probs[:, 1])
        net_auc = auc(fpr, tpr)
        # print('SVM auc {:.4f}'.format(svm_auc))

        print('SVM auc {:.4f}, Net auc {:.4f}'.format(svm_auc, net_auc))

        # print('scores {}'.format(scores))
        # print('pred\n{}'.format(prediction_test))
        # print('{}/10: ACC {:.4f}, AUC {:.4f}'.format(i, accuracy_test, auc_score))
        # aucs.append(auc_score)
        # aucs[i] = auc_score
        # time.sleep(30)
    # mean_auc = np.mean(np.asarray(aucs))
    # print('mean_auc {:.4f}'.format(mean_auc))
    big_aucs = np.asarray(big_aucs)
    print('big_aucs shape {}'.format(big_aucs.shape))
    # avg_aucs = np.mean(big_aucs[[0,1,6,8,9], :], axis=0)
    avg_aucs = np.mean(big_aucs, axis=0)
    # print(avg_aucs)

    # avg_aucs = np.flip(avg_aucs, axis=1)
    # ratios = np.flip(ratios, axis=1)
    avg_aucs = avg_aucs[::-1]
    # ratios = ratios[::-1]
    print(avg_aucs)
    # time.sleep(30)

    best_auc = np.max(avg_aucs)
    best_alpha_index = np.where(avg_aucs == best_auc)
    best_alpha = ratios[best_alpha_index]

    # plt.figure()
    plt.figure(figsize=(7, 4.2))
    # plt.title('KAMP 10-folds')
    plt.plot(ratios, avg_aucs, marker='o', mec='r', mfc='w')
    plt.hlines(y=best_auc, xmin=-0.05, xmax=1.05, colors="grey", linestyles="dashed")
    plt.vlines(x=best_alpha, ymin=0.7, ymax=best_auc, colors="grey", linestyles="dashed")
    plt.xlabel(r'$\alpha$')
    plt.ylabel('AUC Score')
    plt.xlim((0, 1))
    plt.ylim((0.72, 0.84))
    plt.savefig('plots/kamp_ratios/10-folds.jpg')
    # print('max auc {}'.format())
    max_auc = np.max(avg_aucs)
    max_index = np.where(avg_aucs == np.max(avg_aucs))[0][0]
    max_ratio = ratios[max_index]
    print('mean SVM AUC {:.4f}'.format(avg_aucs[-1]))
    print('mean Net AUC {:.4f}'.format(avg_aucs[0]))
    print('Highest AUC {:.4f} with SVM_ratio {}'.format(max_auc, max_ratio))
    time.sleep(30)

def save_diff_for_svm():
    lab_container = np.zeros((1, 3))
    vec_container = np.zeros((1, 120))

    for vid_idx in range(14, 19):
        vid_lab_path = 'const_index/lab/{:02}.txt'.format(vid_idx)
        vid_vec_path = 'data/deg15v1/vid_{:02}.txt'.format(vid_idx)

        vid_lab = np.loadtxt(vid_lab_path)
        vid_lab[:, 0] = vid_idx
        vid_lab[:, 1] = np.linspace(0, vid_lab.shape[0], vid_lab.shape[0], endpoint=False)
        print('vid_lab shape {}'.format(vid_lab.shape))
        # print('vid_lab {}'.format(vid_lab))
        # print('cell_idx shape {}'.format(cell_idx.shape))
        # print('cell_idx {}'.format(cell_idx))
        # time.sleep(30)

        # vid_lab = np.reshape(vid_lab[:, 2], (vid_lab.shape[0], 1))


        vid_vec = np.loadtxt(vid_vec_path)
        print('lab shape {}, vec shape {}'.format(vid_lab.shape, vid_vec.shape))

        lab_container = np.concatenate((lab_container, vid_lab), axis=0)
        vec_container = np.concatenate((vec_container, vid_vec), axis=0)
    print('{}, {}'.format(lab_container.shape, vec_container.shape))

    combined_path = 'data/svm_materials'
    if os.path.isdir(combined_path) == False:
        os.makedirs(combined_path)
    lab_path = path.join(combined_path, 'labels.txt')
    vec_path = path.join(combined_path, 'deg15v1_vecs.txt')

    np.savetxt(lab_path, lab_container[1:, :])
    np.savetxt(vec_path, vec_container[1:, :])
    print('Deg15v1 difference and labels have been saved at {}'.format(combined_path))
    # time.sleep(30)

""" Merge 0 and 1 as 0, convert 2 to 1"""
def merge_class(labels):
    boolean_index = labels == 1
    labels[boolean_index] = 0
    boolean_index = labels == 2
    labels[boolean_index] = 1
    return labels

def train_svm_for_diff():
    combined_path = 'data/svm_materials'
    lab_path = path.join(combined_path, 'labels.txt')
    vec_path = path.join(combined_path, 'deg15v1_vecs.txt')

    vec = np.loadtxt(vec_path)
    lab = np.loadtxt(lab_path)

    # lab = np.reshape(lab, (lab.shape[0], 1))
    all_data = np.concatenate((vec, lab), axis=1)

    all_data[:, -1] = merge_class(all_data[:, -1])
    all_data = all_data[all_data[:, -1].argsort()[::-1]]
    num_cplx = int(np.sum(all_data[:, -1]))
    num_balanced = num_cplx * 2
    all_data = all_data[:num_balanced, :]
    # all_data = np.abs(all_data)
    print(all_data)
    # all_data[:, :120] = sklearn.preprocessing.normalize(all_data[:, :120], norm='l2', axis=1,
    #                                                     copy=True, return_norm=False)
    # unique, counts = np.unique(all_data[:, -1], return_counts=True)
    print('*' * 70)
    print(all_data)
    # time.sleep(30)

    print('{}, {}'.format(vec.shape, lab.shape))
    print('all_data shape {}'.format(all_data.shape))
    print('data_id shape {}'.format(all_data[:, :-1].shape))
    # time.sleep(30)

    index = 0
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    train_accs = []
    test_accs = []
    svms = []
    for train_val_index, test_index in skf.split(all_data[:, :-2], all_data[:, -1]):
        print('*' * 10 + 'fold {}:'.format(index) + '*' * 10)
        train_set = all_data[train_val_index, :]
        test_set = all_data[test_index, :]
        # print('{}/10: train_set shape {}, test_set shape {}'.format(index+1,
        #                                                             train_set.shape,
        #                                                             test_set.shape))
        train_data = train_set[:, :120]
        train_label = train_set[:, -1]

        test_data = test_set[:, :120]
        test_label = test_set[:, -1]

        # train_label = merge_class(train_label)
        # test_label = merge_class(test_label)

        # my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='rbf', C=3000, random_state=50)
        my_svm = sklearn.svm.SVC(gamma='auto', probability=True, kernel='linear', C=100, random_state=50)
        my_svm.fit(train_data, train_label)

        train_result = my_svm.predict(train_data)
        test_result = my_svm.predict(test_data)

        train_acc = (train_label == train_result).sum() / float(train_label.size)
        test_acc = (test_label == test_result).sum() / float(test_label.size)

        print('train_acc {:.4f}, test_acc: {:.4f}'.format(train_acc, test_acc))
        print('test_labels\n{}'.format(test_label))
        print('test_result\n{}'.format(test_result))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        svms.append(my_svm)
        # time.sleep(30)
        # print('train_acc: {:.4f}'.format(train_acc))

        index = index + 1
    avg_train_acc = sum(train_accs) / len(train_accs)
    avg_test_acc = sum(test_accs) / len(test_accs)

    highest_test_acc = max(test_accs)
    best_fold_idx = test_accs.index(highest_test_acc)
    best_svm = svms[best_fold_idx]
    print('avg_train_acc: {:.4f}, avg_test_acc: {:.4f}'.format(avg_train_acc, avg_test_acc))
    print(test_accs)
    print('{} best with test_acc {:.4f}'.format(best_fold_idx, highest_test_acc))

    with open('data/svm_materials/complx_vs_others_SVM.pickle', 'wb') as f:
        pickle.dump(best_svm, f)
        print('Best SVM model saved!'.format())

def test_svm(test_vid_id):
    best_svm = pickle.load(open('data/svm_materials/complx_vs_others_SVM.pickle', 'rb'))

    vid_lab_path = 'const_index/lab/{:02}.txt'.format(test_vid_id)
    vid_vec_path = 'data/deg15v1/vid_{:02}.txt'.format(test_vid_id)

    vid_lab = np.loadtxt(vid_lab_path)
    vid_lab = merge_class(vid_lab[:, 2])
    # vid_lab[:, 0] = test_vid_id
    # vid_lab[:, 1] = np.linspace(0, vid_lab.shape[0], vid_lab.shape[0], endpoint=False)

    vid_vec = np.loadtxt(vid_vec_path)
    print('vid_lab shape {}'.format(vid_lab.shape))
    # print('vid_lab {}'.format(vid_lab))
    print('vid_vec shape {}'.format(vid_vec.shape))

    test_result = best_svm.predict(vid_vec)
    result_acc = (vid_lab == test_result).sum() / float(vid_lab.size)
    print('Vid {} test acc {:.4f}'.format(test_vid_id, result_acc))

if __name__ == '__main__':
    # save_diff_for_svm()

    # train_svm_for_diff()

    test_svm(test_vid_id=18)










