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


    def generate_frames(self, root_path, person_ids: List[int]) -> Iterable[ActionFrameInfo]:
        """Generate frame info for all frames.
        
        Parameters:
            person_ids: persons to include
        """
        person_mark = 0

        #path = os.path.join(root_path, cat_dir)
        #videos = os.listdir(path)

        for root, dirs, files in os.walk(root_path):
            # /media/sally/Elements/1086/Videos & .tif/DX54D_ZVps_8/cam00
            if "cam00" in dirs and "cam01" in dirs:
                print("ROOT ", root)

                for person_id in person_ids:
                    if person_id in root:
                        #files = os.listdir(os.path.join(root, "cam00"))
                        person_mark += 1  # identify all stored frames as belonging to this person + direction
                        dir_path = os.path.join(root, "cam00")
                        filelist = [f for f in os.listdir(dir_path) if f.endswith('.tif')] #os.listdir(dir_path)
                        filelist.sort() 
                        #print(root[-1:])
                        for frame_name in filelist: 
                            yield ActionFrameInfo(
                                file_name=frame_name,
                                file_path=os.path.join(dir_path, frame_name),
                                person_mark=person_mark,
                                material=person_id,
                                folder_index=root[-1:]
                                #category_flag=frame_category_flag
                            )

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

        frames_file_name = []
        frames_person_mark = []  # for each frame in the joint array, mark the person ID
        frames_category = []
        materials = []
        folder_index = []

        # First count total number of frames
        # Do it without creating massive array:
        # all_frame_info = self.generate_frames(path, mode_person_ids)
        tot_num_frames = sum((1 for _ in self.generate_frames(path, mode_person_ids)))
        print(f"Preparing to load {tot_num_frames} video frames.")
        
        # Target array containing ALL RESIZED frames
        data = np.empty((tot_num_frames, self.image_width, self.image_width , 1),
                        dtype=np.int8)  # np.float32

        # Read, resize, and store video frames
        for i, frame in enumerate(self.generate_frames(path, mode_person_ids)):
            #print(frame)
            frame_im = Image.open(frame.file_path).convert('L') # int8 2D array

            # input type must be float32 for default interpolation method cv2.INTER_AREA
            frame_np = np.array(frame_im, dtype=np.float32)  # (1000, 1000) numpy array
            data[i,:,:,0] = (cv2.resize(
                frame_np, (self.image_width,self.image_width))/255).astype(np.int8)

            frames_file_name.append(frame.file_name)
            frames_person_mark.append(frame.person_mark)
            materials.append(frame.material)
            folder_index.append(frame.folder_index)
            #frames_category.append(frame.category_flag)

        #print(frames_file_name)
        #print(materials)
        
        # identify sequences of <seq_len> within the same video
        values, counts = np.unique(materials, return_counts=True)
        print(values, counts)
        values, counts = np.unique(folder_index, return_counts=True)
        print(values, counts)
        indices = counts

        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

