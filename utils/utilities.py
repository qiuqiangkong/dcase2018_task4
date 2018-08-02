import os
import numpy as np
import soundfile
import librosa
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import sed_eval

import torch
from torch.autograd import Variable

import vad
from vad import activity_detection
import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name
    
    
def create_logging(log_dir, filemode):
    
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, "%04d.log" % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, "%04d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
    
    
def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)
        
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    
    return mean, std
   
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    
    
def pad_or_trunc(x, max_len):
    
    if len(x) == max_len:
        return x
    
    elif len(x) > max_len:
        return x[0 : max_len]
        
    else:
        (seq_len, freq_bins) = x.shape
        pad = np.zeros((max_len - seq_len, freq_bins))
        return np.concatenate((x, pad), axis=0)
    

def calculate_auc(target, predict, average='macro'):
    
    return metrics.roc_auc_score(target, predict, average)
    
    
def calculate_ap(target, predict, average='macro'):
    
    return metrics.average_precision_score(target, predict, average)


def calculate_f1_score(target, predict, average='macro'):
    
    import crash
    asdf
    
    return metrics.f1_score(target, predict, average=average)


def read_strong_meta(strong_meta):
    """Read list of events from strong meta. 
    
    Args:
      strong_meta: str, path of strong meta
      
    Returns:
      events_list: list of events
    """
    
    df = pd.read_csv(strong_meta, sep='\t')
    df = pd.DataFrame(df)
    
    events_list = []
    
    for row in df.iterrows():
        event = {'filename': row[1]['filename'], 
                 'onset': row[1]['onset'], 
                 'offset': row[1]['offset'], 
                 'event_label': row[1]['event_label']}
                 
        events_list.append(event)
    
    return events_list
    
    
def calculate_estimated_event_list(audio_names, predictions, frame_wise_probs, 
                                   seconds_per_frame, sed_thres, sed_low_thres):
    """Calculate estimated event list from frame wise probabilites. 
    
    Args:
      audio_names: list of str. 
      predictions: (audios_num, classes_num), value of 0 or 1
      frame_wise_probs: (audios_num, time_steps, classes_num)
      seconds_per_frame: float
      
    Returns:
      estimated_event_list: list of events
    """
    
    ix_to_lb = config.ix_to_lb
    
    estimated_event_list = []
    
    for (n, audio_name) in enumerate(audio_names):
  
        for event_index in predictions[n]:
    
            bgn_fin_pairs = activity_detection(
                frame_wise_probs[n, :, event_index], thres=sed_thres, 
                low_thres=sed_low_thres, n_smooth=1, n_salt=0)
            
            for [bgn, fin] in bgn_fin_pairs:
                
                event = {'filename': audio_name, 
                        'onset': bgn * seconds_per_frame, 
                        'offset': fin * seconds_per_frame, 
                        'event_label': ix_to_lb[event_index]}
    
                estimated_event_list.append(event)
                
    return estimated_event_list
        
        
def read_frame_wise_probs_from_strong_meta(strong_meta, audio_names):
    """Convert strong meta to frame wise probabilites. 
    
    Args:
      strong_meta: str, path of strong meta csv. 
      audio_names: list of str. 
      
    Returns:
      frame_wise_probs: (audios_num, time_steps, classes_num)
    """
    
    seq_len = config.seq_len
    classes_num = len(config.labels)
    lb_to_ix = config.lb_to_ix
    hop_samples = config.window_size - config.overlap
    sample_rate = config.sample_rate
    
    seconds_per_frame = hop_samples / float(sample_rate)
    
    # Read strong meta
    df = pd.read_csv(strong_meta, sep='\t')
    df = pd.DataFrame(df)
    
    frame_wise_probs = np.zeros((len(audio_names), seq_len, classes_num))
    
    for (n, audio_name) in enumerate(audio_names):
        indexes = df.index[df['filename'] == audio_name].tolist()
        
        for index in indexes:
            event_label = df.iloc[index]['event_label']
            onset = df.iloc[index]['onset']
            offset = df.iloc[index]['offset']
            frame_wise_probs[n, int(onset / seconds_per_frame) : 
                int(offset / seconds_per_frame), lb_to_ix[event_label]] = 1
        
    return frame_wise_probs
            
            
def get_binary_predictions(probs, thres):
    """Get binary predictions from probability. 
    
    Args:
      probs: (audios_num, classes_num), value between 0 and 1
      thres: float
      
    Returns:
      predictions, (audios_num, classes_num), value of 0 or 1
    """
    
    predictions = []
    
    for prob in probs:
        
        prediction = np.where(prob > thres)[0]
        
        if len(prediction) == 0:
            prediction = np.array([np.argmax(prob)])
        
        predictions.append(prediction)
        
    return np.array(predictions)
    
    
def event_based_evaluation(reference_event_list, estimated_event_list):
    """ Calculate sed_eval event based metric for challenge
        Parameters
        ----------
        reference_event_list : MetaDataContainer, list of referenced events
        estimated_event_list : MetaDataContainer, list of estimated events
        Return
        ------
        event_based_metric : EventBasedMetrics
        """

    files = {}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        # event_label_list=reference_event_list.unique_event_labels,
        event_label_list = config.labels, 
        t_collar=0.200,
        percentage_of_length=0.2,
    )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        # events = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                # events.append(event.event_label)
        estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return event_based_metric
    
    
def write_testing_data_submission_csv(estimated_event_list, submission_path):
    
    # Write out submission file
    f = open(submission_path, 'w')

    for (n, event) in enumerate(estimated_event_list):
        
        f.write('{}\t{}\t{}\t{}\n'.format(
            event['filename'], event['onset'], event['offset'], 
            event['event_label']))

    f.close()
    
    print('Write out submission file to {}'.format(submission_path))