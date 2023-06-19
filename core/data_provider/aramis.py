__author__ = 'gaozhifeng'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from typing import Iterable, List
from dataclasses import dataclass
from collections import Counter

from tqdm import tqdm
from threading import Thread
from typing import Union, List
import time


logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        print("Minipatch size ", self.minibatch_size)
        self.image_width = input_param['image_width']
        self.image_height = input_param['image_height']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        # Create batch of N videos of length L, i.e. shape = (N, L, w, h, c)
        # where w x h is the resolution and c the number of color channels
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 1)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


@dataclass
class ActionFrameInfo:
    file_name: str
    file_path: str
    person_mark: int
    material: int
    folder_index: int
    #category_flag: int


class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']  # path to parent folder containing category dirs
        self.image_width = input_param['image_width']
        self.image_height = input_param['image_height']

        # Hard coded training and test persons (prevent same person occurring in train - test set)
        self.train_material = ['1086']#['1086', '1439']
        self.test_material = ['1086']

        self.input_param = input_param
        self.seq_len = input_param['seq_length']


    def load_data(self, paths, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        if mode == 'train':
            mode_person_ids = self.train_material
        elif mode == 'test':
            mode_person_ids = self.test_material
        else:
            raise Exception("Unexpected mode: " + mode)
        print('begin load data' + str(path))

        indices = []

        npz_files = []

        if isinstance(path, str) and path.endswith(".npz"):
            npz_files.append(path)
            print("added ", path)
        else:
            print("searching ")
            #for folder_path in path:
                #print(folder_path)
            for root, dirs, files in os.walk(path):
                #print(files)
                for file in files:
                    if file.endswith(".npz") and not file.startswith("._"):
                        npz_files.append(os.path.join(root, file))

        print("found ", npz_files)
        tot_num_frames = 0

        for file in tqdm(npz_files):
            X = np.load(file, allow_pickle=True)["arr_0"]
            tot_num_frames += len(X)

        """ # original data loading
        frame_im = Image.open(frame.file_path).convert('L') # int8 2D array

        # input type must be float32 for default interpolation method cv2.INTER_AREA
        frame_np = np.array(frame_im, dtype=np.float32)  # (1000, 1000) numpy array
        data[i,:,:,0] = (cv2.resize(
        frame_np, (self.image_width,self.image_width))/255).astype(np.int8)
        """

        print("total frames ", tot_num_frames)

        data = np.empty((tot_num_frames, self.image_width, self.image_width , 1),
                        dtype=np.int8)
        
        index = 0
        data_filler = 0
        for file in tqdm(npz_files):
            X = np.load(file, allow_pickle=True)["arr_0"]
            for i in range(len(X)):
                indices.append(index)
                X1, X2 = np.split(X[i], 2, axis=0)
                data[data_filler,:,:,0] = X1
                data_filler += 1

            index +=1
            #print("index ", index)

        values, counts = np.unique(indices, return_counts=True)
        print(values, counts)

        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(counts)) + " sequences")

        return data, counts

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)


