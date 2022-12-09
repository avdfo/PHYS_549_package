import os, inspect, glob

import numpy as np
from sklearn.utils import shuffle

class DataSet(object):

    def __init__(self):

        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../dataset"

        self.list_train_lr = self.sorted_list(os.path.join(self.data_path, "train_lr", "*.npy"))
        self.list_train_hr = self.sorted_list(os.path.join(self.data_path, "train_hr", "*.npy"))

        self.list_validation_lr = self.sorted_list(os.path.join(self.data_path, "validation_lr", "*.npy"))
        self.list_validation_hr = self.sorted_list(os.path.join(self.data_path, "validation_hr", "*.npy"))

        self.list_test_lr = self.sorted_list(os.path.join(self.data_path, "test_lr", "*.npy"))
        self.list_test_hr = self.sorted_list(os.path.join(self.data_path, "test_hr", "*.npy"))

        self.list_test_exp = self.sorted_list(os.path.join(self.data_path, "test_exp", "*.npy"))

        self.amount_tr = len(self.list_train_lr)
        self.amount_val = len(self.list_validation_lr)
        self.amount_te = len(self.list_test_lr)
        self.amount_te_exp = len(self.list_test_exp)

        self.idx_tr = 0
        self.idx_val = 0
        self.idx_te = 0
        self.idx_te_exp = 0

    def sorted_list(self, path):
        tmplist = glob.glob(path)
        tmplist.sort()

        return tmplist

    def next_train(self, batch_size=1):

        data = np.zeros((0, 1, 1, 1))
        label = np.zeros((0, 1, 1, 1))
        terminator = False

        while(True):
            data_tmp = np.expand_dims(np.load(self.list_train_lr[self.idx_tr]), axis=0)
            label_tmp = np.expand_dims(np.load(self.list_train_hr[self.idx_tr]), axis=0)

            if(len(data_tmp.shape) < 4):
                data_tmp = np.expand_dims(data_tmp, axis=3)
                label_tmp = np.expand_dims(label_tmp, axis=3)

            if(data.shape[0] == 0):
                data = data_tmp
                label = label_tmp
            else:
                if((data.shape[1] == data_tmp.shape[1]) and (data.shape[2] == data_tmp.shape[2]) and (data.shape[3] == data_tmp.shape[3])):
                    data = np.append(data, data_tmp, axis=0)
                    label = np.append(label, label_tmp, axis=0)

            self.idx_tr += 1
            if(self.idx_tr >= self.amount_tr):
                self.list_train_lr, self.list_train_hr = shuffle(self.list_train_lr, self.list_train_hr)
                self.idx_tr = 0
                terminator = True
                break
            elif(data.shape[0] == batch_size): break
            else: pass

        return data, label, terminator

    def next_val(self, batch_size_val=1):
        data = np.zeros((0, 1, 1, 1))
        label = np.zeros((0, 1, 1, 1))

        for i in range(batch_size_val):
            if self.idx_val >= self.amount_val:
                raise Exception('Need more validation data to match the batch size ratio between training and validation datasets')

            data_tmp = np.expand_dims(np.load(self.list_validation_lr[self.idx_val]), axis=0)
            label_tmp = np.expand_dims(np.load(self.list_validation_hr[self.idx_val]), axis=0)

            if (len(data_tmp.shape) < 4):
                data_tmp = np.expand_dims(data_tmp, axis=3)
                label_tmp = np.expand_dims(label_tmp, axis=3)

            if (data.shape[0] == 0):
                data = data_tmp
                label = label_tmp

            else:
                if((data.shape[1] == data_tmp.shape[1]) and (data.shape[2] == data_tmp.shape[2]) and (data.shape[3] == data_tmp.shape[3])):
                    data = np.append(data, data_tmp, axis=0)
                    label = np.append(label, label_tmp, axis=0)

            self.idx_val += 1

        return data, label

    def next_test(self):

        data = np.expand_dims(np.load(self.list_test_lr[self.idx_te]), axis=0)
        label = np.expand_dims(np.load(self.list_test_hr[self.idx_te]), axis=0)

        if(len(data.shape) < 4):
            data = np.expand_dims(data, axis=3)
            label = np.expand_dims(label, axis=3)

        self.idx_te += 1

        if(self.idx_te > self.amount_te):
            self.idx_te = 0
            return None, None
        else: return data, label

    def next_test_exp(self):

        data = np.expand_dims(np.load(self.list_test_exp[self.idx_te_exp]), axis=0)

        if (len(data.shape) < 4):
            data = np.expand_dims(data, axis=3)

        self.idx_te_exp += 1

        if (self.idx_te_exp > self.amount_te_exp):
            self.idx_te_exp = 0
            return None
        else: return data
