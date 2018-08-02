import os
import sys
import numpy as np
import h5py
import time
import logging
from utilities import calculate_scalar, scale
import config


class DataGenerator(object):
    
    def __init__(self, train_hdf5_path, validate_hdf5_path, batch_size, 
                 validate, seed=1234):
        """Data generator. 
        
        Args:
          train_hdf5_path: str, path of train hdf5 file
          validate_hdf5_path: str, path of validate hdf5 path
          batch_size: int
          validate: bool
          seed: int
        """
        
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        lb_to_ix = config.lb_to_ix

        self.batch_size = batch_size
        self.validate = validate

        # Load data
        load_time = time.time()
        
        hf = h5py.File(train_hdf5_path, 'r')
        self.train_audio_names = np.array([s.decode() for s in hf['audio_name'][:]])
        self.train_x = hf['feature'][:]
        self.train_y = hf['target'][:]
        hf.close()

        hf = h5py.File(validate_hdf5_path, 'r')
        self.validate_audio_names = np.array([s.decode() for s in hf['audio_name']])
        self.validate_x = hf['feature'][:]
        self.validate_y = hf['target'][:]
        hf.close()    
        
        logging.info('Loading data time: {:.3f} s'
            ''.format(time.time() - load_time))

        # Get train & validate audio indexes
        self.audio_names = np.concatenate(
            (self.train_audio_names, self.validate_audio_names), axis=0)
        
        self.x = np.concatenate((self.train_x, self.validate_x), axis=0)
        self.y = np.concatenate((self.train_y, self.validate_y), axis=0)
        
        if validate:
            
            self.train_audio_indexes = np.arange(len(self.train_audio_names))
            
            self.validate_audio_indexes = np.arange(
                len(self.train_audio_names), 
                len(self.train_audio_names) + len(self.validate_audio_names))
            
        else:
            self.train_audio_indexes = np.arange(len(self.audio_names))
            self.validate_audio_indexes = np.array([])
        
        logging.info("Training audios: {}".format(
            len(self.train_audio_indexes)))
        
        logging.info("Validation audios: {}".format(
            len(self.validate_audio_indexes)))
        
        # Calculate scalar
        (self.mean, self.std) = calculate_scalar(
            self.x[self.train_audio_indexes])
        
    def generate_train(self):
        """Generate mini-batch data for training. 
        """
        
        batch_size = self.batch_size
        indexes = np.array(self.train_audio_indexes)
        samples = len(indexes)
        
        self.random_state.shuffle(indexes)
        
        iteration = 0
        pointer = 0
        
        while True:
            
            # Reset pointer
            if pointer >= samples:
                pointer = 0
                self.random_state.shuffle(indexes)
            
            # Get batch indexes
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            iteration += 1
            
            batch_x = self.x[batch_indexes]
            batch_y = self.y[batch_indexes]
            
            # Transform data
            batch_x = self.transform(batch_x)
            batch_y = batch_y.astype(np.float32)
            
            yield batch_x, batch_y
        
    def generate_validate(self, data_type, shuffle=False, max_iteration=None):
        """Generate mini-batch data for validation. 
        
        Args:
          data_type: 'train' | 'validate'
          shuffle: bool
          max_iteration: int, maximum iteration for speed up validation
        """
    
        batch_size = self.batch_size
        
        if data_type == 'train':
            indexes = np.array(self.train_audio_indexes)
            
        elif data_type == 'validate':
            indexes = np.array(self.validate_audio_indexes)
            
        else:
            raise Exception("Invalid data_type!")
            
        audios_num = len(indexes)
        
        if shuffle:
            self.validate_random_state.shuffle(indexes)
        
        iteration = 0
        pointer = 0
        
        while True:
            
            if iteration == max_iteration:
                break
            
            if pointer >= audios_num:
                break
            
            # Get batch indexes
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            iteration += 1
  
            batch_x = self.x[batch_indexes]
            batch_y = self.y[batch_indexes]
            batch_audio_names = self.audio_names[batch_indexes]
            
            # Transform data
            batch_x = self.transform(batch_x)
            batch_y = batch_y.astype(np.float32)
            
            yield batch_x, batch_y, batch_audio_names
            
            
    def transform(self, x):
        """Transform data. 
        
        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
          
        Returns:
          Transformed data. 
        """

        return scale(x, self.mean, self.std)
            
            
class TestDataGenerator(DataGenerator):
    
    def __init__(self, train_hdf5_path, validate_hdf5_path, eval_hdf5_path,
                 batch_size):
        """Test data generator. 
        
        Args:
          train_hdf5_path: str, path of training hdf5 file
          validate_hdf5_path, str, path of validation hdf5
          eval_hdf5_path: str, path of evaluation hdf5 file
          batch_size: int
        """
        
        super(TestDataGenerator, self).__init__(
            train_hdf5_path=train_hdf5_path, 
            validate_hdf5_path=validate_hdf5_path, 
            batch_size=batch_size, 
            validate=False)
            
        # Load data
        load_time = time.time()
        hf = h5py.File(eval_hdf5_path, 'r')
        
        self.eval_audio_names = np.array(
            [name.decode() for name in hf['audio_name'][:]])
        
        self.eval_x = hf['feature'][:]
        
        logging.info("Load data time: {}".format(time.time() - load_time))
        
    def generate_eval(self):
        
        audios_num = len(self.eval_audio_names)
        audio_indexes = np.arange(audios_num)
        batch_size = self.batch_size
        
        pointer = 0
        
        while True:

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
                
            pointer += batch_size

            batch_x = self.eval_x[batch_audio_indexes]
            batch_audio_names = self.eval_audio_names[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_audio_names