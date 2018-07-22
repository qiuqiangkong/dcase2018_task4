import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy
import argparse
import sys
import soundfile
import numpy as np
import librosa
import h5py
import time
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from utilities import (read_audio, create_folder, calculate_scalar, 
                           create_logging, get_filename, pad_or_trunc)
import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
    
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
            
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x


def calculate_logmel(audio_path, sample_rate, feature_extractor, seq_len):
    
    # Read audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
    
    # Normalize energy
    audio /= np.max(np.abs(audio))
    
    # Extract feature
    feature = feature_extractor.transform(audio)
    
    feature = pad_or_trunc(feature, seq_len)
    
    return feature
    
    
def write_strong_meta_to_weak_meta(meta_csv, formated_csv):
    
    create_folder(os.path.dirname(formated_csv))
    
    # Read meta csv
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)
    
    dict = {}
    
    for row in df.iterrows():
        
        audio_name = row[1]['filename']
        
        event_label = row[1]['event_label']
        
        if audio_name not in dict.keys():
            dict[audio_name] = [event_label]
        
        else:
            if event_label not in dict[audio_name]:
                dict[audio_name].append(event_label)
        
    # Write weak labels to csv
    f = open(formated_csv, 'w')
        
    f.write('{}\t{}\n'.format('filename', 'event_labels'))
        
    for key in dict.keys():
        f.write('{}\t{}\n'.format(key, ','.join(dict[key])))
        
    f.close()
    
    print('Write formated_csv to {}'.format(formated_csv))
    
    
def read_meta(meta_csv):
    
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)
    
    audio_names = df['filename']
    
    if 'event_labels' in df.keys():
        
        event_labels = df['event_labels']
        event_labels = [s.split(',') for s in event_labels]
        return audio_names, event_labels
        
    else:
        return (audio_names,)
    

def logmel(args, audios_dir, meta_csv, hdf5_path):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    labels = config.labels
    classes_num = len(labels)
    lb_to_ix = config.lb_to_ix
    
    data_type = 'train'
    feature_type = 'logmel'

    # Paths
    create_folder(os.path.dirname(hdf5_path))
    

    # Read meta csv
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)
    
    has_labels = 'event_labels' in df.keys()
    has_onset = 'onset' in df.keys()
    has_offset = 'offset' in df.keys()
    
    tuple = read_meta(meta_csv)
    
    if len(tuple) == 1:
        (audio_names,) = tuple
        has_event_labels = False
        
    elif len(tuple) == 2:
        (audio_names, event_labels) = tuple
        has_event_labels = True
        
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    # Create hdf5 file
    begin_time = time.time()
    hf = h5py.File(hdf5_path, 'w')
    
    hf.create_dataset(
        name='feature', 
        shape=(0, seq_len, mel_bins), 
        maxshape=(None, seq_len, mel_bins), 
        dtype=np.float32)
    
    for (n, audio_name) in enumerate(audio_names):
        
        print(n, audio_name)
        
        audio_path = os.path.join(audios_dir, audio_name)
        
        # Extract feature
        feature = calculate_logmel(audio_path=audio_path, 
                                   sample_rate=sample_rate, 
                                   feature_extractor=feature_extractor, 
                                   seq_len=seq_len)
        '''(seq_len, mel_bins)'''
        
        print(feature.shape)
        
        hf['feature'].resize((n + 1, seq_len, mel_bins))
        hf['feature'][n] = feature
        
        # Plot log Mel for debug
        if False:
            plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
            plt.show()
    
    # Write out meta to hdf5 file
    hf.create_dataset(name='audio_name', 
                      data=[s.encode() for s in audio_names], 
                      dtype='S50')
        
    hf.create_dataset(name='labels', 
                      data=[s.encode() for s in labels], 
                      dtype='S30')
        
    if has_event_labels:
        
        target = np.zeros((len(audio_names), classes_num), dtype=np.int32)
        
        for n in range(len(audio_names)):
            for lb in event_labels[n]:
                target[n][lb_to_ix[lb]] = 1
        
        hf.create_dataset(name='target', 
                          data=target, 
                          dtype=np.int32)

    hf.close()
          
    print("Write out to {}".format(hdf5_path))
    print("Time: {} s".format(time.time() - begin_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--data_type', type=str, 
        choices=['dev_train_weak', 'dev_train_unlabel_out_of_domain', 'dev_train_unlabel_in_domain', 'dev_test', 'eval'], required=True)
    
    args = parser.parse_args()
    
    logs_dir = os.path.join(args.workspace, 'logs', get_filename(__file__))
    logging = create_logging(logs_dir, filemode='w')
    
    if args.mode == 'logmel':
        
        if args.data_type == 'dev_train_weak':
            logmel(args, 
                audios_dir=os.path.join(args.dataset_dir, 'audio', 'train', 'weak'), 
                meta_csv=os.path.join(args.dataset_dir, 'metadata', 'train', 'weak.csv'), 
                hdf5_path=os.path.join(args.workspace, 'features', 'logmel', 'train', 'weak.h5'))
        
        if args.data_type == 'dev_train_unlabel_in_domain':
            logmel(args, 
                audios_dir=os.path.join(args.dataset_dir, 'audio', 'train', 'unlabel_in_domain'), 
                meta_csv=os.path.join(args.dataset_dir, 'metadata', 'train', 'unlabel_in_domain.csv'), 
                hdf5_path=os.path.join(args.workspace, 'features', 'logmel', 'train', 'unlabel_in_domain.h5'))
                
        if args.data_type == 'dev_train_unlabel_out_of_domain':
            logmel(args, 
                audios_dir=os.path.join(args.dataset_dir, 'audio', 'train', 'unlabel_out_of_domain'), 
                meta_csv=os.path.join(args.dataset_dir, 'metadata', 'train', 'unlabel_out_of_domain.csv'), 
                hdf5_path=os.path.join(args.workspace, 'features', 'logmel', 'train', 'unlabel_out_of_domain.h5'))
        
        elif args.data_type == 'dev_test':
            
            formated_csv = os.path.join(args.workspace, 'formated_meta', 'test', 'test.csv')
            
            write_strong_meta_to_weak_meta(
                meta_csv=os.path.join(args.dataset_dir, 'metadata', 'test', 'test.csv'), 
                formated_csv=formated_csv)
            
            logmel(args, 
                audios_dir=os.path.join(args.dataset_dir, 'audio', 'test'), 
                meta_csv=formated_csv, 
                hdf5_path=os.path.join(args.workspace, 'features', 'logmel', 'test', 'test.h5'))
                
        elif args.data_type == 'eval':
            logmel(args, 
                audios_dir=os.path.join(args.dataset_dir, 'audio', 'eval'), 
                meta_csv=os.path.join(args.dataset_dir, 'metadata', 'eval', 'eval.csv'), 
                hdf5_path=os.path.join(args.workspace, 'features', 'logmel', 'eval', 'eval.h5'))
            
        else:
            raise Exception('Incorrect argument!')