import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging, 
                       calculate_auc, calculate_ap, read_strong_meta, 
                       calculate_estimated_event_list, 
                       read_frame_wise_probs_from_strong_meta, 
                       get_binary_predictions, event_based_evaluation, 
                       write_submission)
from models_pytorch import move_data_to_gpu, init_layer, init_bn, BaselineCnn
import config


# Hyper parameters
at_thres = 0.8

sed = False

if sed:
    sed_thres = 0.8
    sed_low_thres = 0.2
else:
    sed_thres = 0.
    sed_low_thres = 0.


def evaluate(model, generator, data_type, max_iteration, cuda):

    generate_func = generator.generate_validate(data_type=data_type, 
                                                max_iteration=max_iteration)

    # Inference
    dict = forward(
        model=model,
        generate_func=generate_func,
        cuda=cuda,
        return_inputs=False, 
        return_targets=True, 
        return_frame_wise_probs=False)
        
    outputs = dict['outputs']
    targets = dict['targets']
    
    # Calculate metrics
    loss = F.binary_cross_entropy(torch.Tensor(outputs), 
                                  torch.Tensor(targets)).numpy()
    loss = float(loss)

    map = calculate_ap(targets, outputs)

    mauc = calculate_auc(targets, outputs)

    # return map, mauc, loss
    return map, mauc, loss
    

def forward(model, generate_func, cuda, return_inputs, return_targets, 
            return_frame_wise_probs):
    """Forward data to a model. 
    
    model: object. 
    generator_func: function. 
    return_bottleneck: bool. 
    cuda: bool. 
    """
    
    outputs = []
    audio_names = []

    if return_inputs: 
        inputs = []
    
    if return_targets:
        targets = []
        
    if return_frame_wise_probs:
        frame_wise_probs = []
    
    iteration = 0

    # Evaluate on mini-batch
    for data in generate_func:
        
        if return_targets:
            (batch_x, batch_y, batch_audio_names) = data
            
        else:
            (batch_x, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        
        if return_frame_wise_probs:
            
            (batch_output, batch_frame_wise_probs) = model(
                batch_x, return_frame_wise_probs=True)
            
        else:
            batch_output = model(batch_x, return_frame_wise_probs=False)
        
        # Append data    
        outputs.append(batch_output.data.cpu().numpy())
        
        audio_names.append(batch_audio_names)
        
        if return_inputs:
            inputs.append(batch_x)
            
        if return_targets:
            targets.append(batch_y)
        
        if return_frame_wise_probs:
            frame_wise_probs.append(batch_frame_wise_probs.data.cpu().numpy())
        
        iteration += 1

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['outputs'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_names'] = audio_names
    
    if return_inputs:
        inputs = np.concatenate(inputs, axis=0)
        dict['inputs'] = inputs
        
    if return_targets:
        targets = np.concatenate(targets, axis=0)
        dict['targets'] = targets
        
    if return_frame_wise_probs:
        frame_wise_probs = np.concatenate(frame_wise_probs, axis=0)
        dict['frame_wise_probs'] = frame_wise_probs

    return dict

def train(args):
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    validate = args.validate
    cuda = args.cuda
    filename = args.filename
    classes_num = len(config.labels)
    
    batch_size = 64

    # Paths
    train_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'train', 
                                   'weak.h5')
                                   
    validate_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'test', 
                                      'test.h5')
    
    if validate:
        validation_csv = os.path.join(dataset_dir, 'metadata', 'test', 
                                      'test.csv')
        
        models_dir = os.path.join(workspace, 'models', filename, 
                                'validate={}'.format(validate))
        
    else:
        validation_csv = None
        models_dir = os.path.join(workspace, 'models', filename, 
                                  'validate={}'.format(validate))
                              
    create_folder(models_dir)

    # Model
    model = BaselineCnn(classes_num)

    if cuda:
        model.cuda()

    generator = DataGenerator(train_hdf5_path=train_hdf5_path,
                              validate_hdf5_path=validate_hdf5_path, 
                              batch_size=batch_size,
                              validate=validate)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    iteration = 0
    train_bgn_time = time.time()

    # Train on mini-batch
    for (batch_x, batch_y) in generator.generate_train():

        # Evaluate
        if iteration % 100 == 0 and iteration > 0:

            train_fin_time = time.time()
            
            
            (tr_ap, tr_auc, tr_loss) = evaluate(model=model,
                                                 generator=generator,
                                                 data_type='train',
                                                 max_iteration=None,
                                                 cuda=cuda)
            

            if validate:
                (va_ap, va_auc, va_loss) = evaluate(model=model,
                                                    generator=generator,
                                                    data_type='validate',
                                                    max_iteration=None,
                                                    cuda=cuda)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            # Print info
            logging.info(
                "iteration: {}, train time: {:.3f} s, validate time: {:.3f} s".format(
                    iteration, train_time, validate_time))
                    
            logging.info(
                "tr_ap: {:.3f}, tr_auc: {:.3f}, tr_loss: {:.3f}".format(
                    tr_ap, tr_auc, tr_loss))
            
            if validate:
                logging.info(
                    "va_ap: {:.3f}, va_auc: {:.3f}, va_loss: {:.3f}".format(
                        va_ap, va_auc, va_loss))
                    
            logging.info("")

            train_bgn_time = time.time()

        # Move data to gpu
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        # Train
        model.train()
        output = model(batch_x)
        loss = F.binary_cross_entropy(output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iteration += 1

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info("Model saved to {}".format(save_out_path))
            
        # Reduce learning rate
        if iteration % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
                
        # Stop learning
        if iteration == 10000:
            break


def inference_validation(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    visualize = args.visualize

    validate = True
    batch_size = 64
    labels = config.labels
    classes_num = len(config.labels)
    hop_samples = config.window_size - config.overlap
    sample_rate = config.sample_rate

    # Paths
    train_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'train', 
                                   'weak.h5')
                                   
    validate_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'test', 
                                      'test.h5')

    validation_csv = os.path.join(dataset_dir, 'metadata', 'test', 'test.csv')

    model_path = os.path.join(workspace, 'models', filename, 
                              'validate={}'.format(validate), 
                              'md_{}_iters.tar'.format(iteration))

    # Load model
    model = BaselineCnn(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    generator = DataGenerator(train_hdf5_path=train_hdf5_path,
                              validate_hdf5_path=validate_hdf5_path, 
                              batch_size=batch_size,
                              validate=validate)

    generate_func = generator.generate_validate(data_type='validate')

    # Inference
    dict = forward(model=model,
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_inputs=True, 
                   return_targets=True, 
                   return_frame_wise_probs=True)

    inputs = dict['inputs']
    audio_names = dict['audio_names']
    outputs = dict['outputs']
    targets = dict['targets']
    frame_wise_probs = dict['frame_wise_probs']

    # Calculate metrics
    ap = calculate_ap(targets, outputs, average=None)
    auc = calculate_auc(targets, outputs, average=None)
    
    print('Audio Tagging result:')
    print('{:<30}{}\t{}'.format('', 'mAP', 'mAUC'))
    print('------------------------------------')
    for (n, label) in enumerate(labels):
        print('{:<30}{:.3f}\t{:.3f}'.format(label, ap[n], auc[n]))
    print('------------------------------------')
    print('{:<30}{:.3f}\t{:.3f}'.format('Average', np.mean(ap), np.mean(auc)))
    
    # Get audio tagging binary predictions
    predictions = get_binary_predictions(outputs, thres=at_thres)
    
    # Get estimated event list
    seconds_per_frame = hop_samples / float(sample_rate)    
    
    estimated_event_list = calculate_estimated_event_list(
        audio_names, predictions, frame_wise_probs, seconds_per_frame, 
        sed_thres, sed_low_thres)

    # Read reference event list
    reference_event_list = read_strong_meta(validation_csv)
    
    # Evaluate sound event detection metrics
    event_based_metric = event_based_evaluation(reference_event_list, 
                                                estimated_event_list)
    
    print(event_based_metric)
    
    # Plot sound event detection ground truth & prediction 
    reference_frame_wise_probs = read_frame_wise_probs_from_strong_meta(
        validation_csv, audio_names)
    
    # Plot
    if visualize:
        np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)
        for n in range(len(audio_names)):
            print(audio_names[n])
            print(targets[n])
            print(outputs[n])
            print(config.labels)
            fig, axs = plt.subplots(6,2, sharex=True)
            axs[0, 0].matshow(inputs[n].T, origin='lower', aspect='auto', cmap='jet')
            for k in range(10):
                axs[k//2+1, k%2].plot(frame_wise_probs[n, :, k])
                axs[k//2+1, k%2].plot(reference_frame_wise_probs[n, :, k], 'r')
                axs[k//2+1, k%2].set_ylim(0, 1.01)
            
            plt.show()
        
        
def inference_testing_data(args):
    
    # Arugments & parameters
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    
    validate = False
    batch_size = 64
    labels = config.labels
    classes_num = len(config.labels)
    hop_samples = config.window_size - config.overlap
    sample_rate = config.sample_rate
    seconds_per_frame = hop_samples / float(sample_rate)    

    # Paths
    train_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'train', 
                                   'weak.h5')
                                   
    validate_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'test', 
                                      'test.h5')
                                 
    eval_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'eval', 
                                  'eval.h5')
                                 
    model_path = os.path.join(workspace, 'models', filename, 
                              'validate={}'.format(validate), 
                              'md_{}_iters.tar'.format(iteration))
                              
    submission_path = os.path.join(workspace, 'submissions', filename, 
                                   'iteration={}'.format(iteration), 
                                   'submission.csv')
                                   
    create_folder(os.path.dirname(submission_path))
    
    # Load model
    model = BaselineCnn(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    generator = TestDataGenerator(train_hdf5_path=train_hdf5_path,
                                  validate_hdf5_path=validate_hdf5_path, 
                                  eval_hdf5_path=eval_hdf5_path, 
                                  batch_size=batch_size)

    generate_func = generator.generate_eval()

    # Inference
    dict = forward(model=model,
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_inputs=True, 
                   return_targets=False, 
                   return_frame_wise_probs=True)

    inputs = dict['inputs']
    audio_names = dict['audio_names']
    outputs = dict['outputs']
    frame_wise_probs = dict['frame_wise_probs']
    
    # Get audio tagging binary predictions
    predictions = get_binary_predictions(outputs, thres=at_thres)
    
    # Get estimated event list
    estimated_event_list = calculate_estimated_event_list(
        audio_names, predictions, frame_wise_probs, seconds_per_frame, 
        sed_thres, sed_low_thres)

    write_submission(estimated_event_list, submission_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--iteration', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False)
    
    parser_inference_testing_data = subparsers.add_parser('inference_testing_data')
    parser_inference_testing_data.add_argument('--workspace', type=str, required=True)
    parser_inference_testing_data.add_argument('--iteration', type=int, required=True)
    parser_inference_testing_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)
    
    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_folder(os.path.dirname(logs_dir))
    logging = create_logging(logs_dir, filemode='w')

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)
        
    elif args.mode == 'inference_testing_data':
        inference_testing_data(args)

    else:
        raise Exception('In correct argument!')