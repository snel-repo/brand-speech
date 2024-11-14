import copy
import logging
import os
import pickle
import random
import sys
import time
from glob import glob
from pathlib import Path
import re
from g2p_en import G2p

import redis
import numpy as np
import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader
from omegaconf import OmegaConf

from speech_utils.utils.gaussSmooth import gaussSmooth

import sys
sys.path.append('../brand-modules/brand-speech/gru_phoneme_online_trainer')
from train_decoder import brainToText_decoder

PHONE_DEF_SIL = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', 'SIL'
]
SIL_DEF = ['SIL']


def load_model(init_model_dir, ckpt_idx, gpu_number):
    cwd = os.getcwd()
    os.chdir(init_model_dir)

    # Load model config and set which checkpoint to load
    args = OmegaConf.load(os.path.join(init_model_dir, 'args.yaml'))
    # To make sure that the model sees the right training_log directory
    args['outputDir'] = init_model_dir
    args['loadDir'] = './'
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = ckpt_idx
    model_sessions = args['dataset']['sessions']
    args['dataset']['sessions'] = []  # Do not load any dataset
    args['gpuNumber'] = str(gpu_number)

    # Initialize model
    tf.compat.v1.reset_default_graph()
    nsd = brainToText_decoder(args)

    os.chdir(cwd)

    return nsd, model_sessions


def add_input_layers(model, existing_layers, requested_layers):
    logging.info(f'existing layers: {existing_layers}')
    logging.info(f'requested layers: {requested_layers}')
    layers_to_add = [l for l in requested_layers if l not in existing_layers]
    logging.info(f'Adding input layer {layers_to_add}')
    last_layer = model.inputLayers[-1]
    for l in layers_to_add:
        # input_dim = model.args['dataset']['nInputFeatures']
        # new_layer = tf.keras.layers.Dense(model.args['model']['inputLayerSize'],
        #                                   kernel_regularizer=tf.keras.regularizers.L2(model.args['model']['weightReg']))
        # new_layer.build(input_shape=[input_dim])

        inputModel = tf.keras.Sequential()
        inputModel.add(tf.keras.Input(shape=(None, model.args['dataset']['nInputFeatures'])))
        inputModel.add(tf.keras.layers.Dense(model.args['model']['inputLayerSize'],
                                                activation = model.args['model']['inputNetwork']['activation'],
                                                kernel_initializer = tf.keras.initializers.identity(),
                                                kernel_regularizer = tf.keras.regularizers.L2(model.args['model']['weightReg'])))
        inputModel.add(tf.keras.layers.Dropout(rate = model.args['model']['inputNetwork']['dropout']))

        # Copy weights
        from_layer = model.inputLayers[l] if l in existing_layers else last_layer
        for vf, vt in zip(from_layer.variables, inputModel.variables):
            vt.assign(vf)

        logging.info(f'Copied input layer {l} weights from {existing_layers[-1]}')

        model.inputLayers.append(inputModel)



def data_generator(data_buffer):
    for k, data in data_buffer.items():
        for i, d in enumerate(data):
            if 'newClassSignal' in d:
                del d['newClassSignal']
            if 'ceMask' in d:
                del d['ceMask']
            d['layerIdx'] = k

            yield d


def load_data(rnn_args, data_dir, normalize=True):
    tfrecords = []
    tfrecords.extend(glob(os.path.join(data_dir, 'train', '*.tfrecord')))
    logging.info(f'Loading the following tfrecords:')
    logging.info(tfrecords)
    dataset = None
    data_buffer = []
    if len(tfrecords) > 0:
        dataset = tf.data.TFRecordDataset(filenames=tfrecords)

        # this is the encoding for the tfrecord data files
        datasetFeatures = {
            "inputFeatures": tf.io.FixedLenSequenceFeature([rnn_args['dataset']['nInputFeatures']], tf.float32, allow_missing=True),
            #"classLabelsOneHot": tf.io.FixedLenSequenceFeature([self.nClasses+1], tf.float32, allow_missing=True),
            "newClassSignal": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "ceMask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "seqClassIDs": tf.io.FixedLenFeature((rnn_args['dataset']['maxSeqElements']), tf.int64),
            "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
            "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
            "transcription": tf.io.FixedLenFeature((rnn_args['dataset']['maxSeqElements']), tf.int64)
        }

        # use tfrecord encoding to parse data into dictionary
        def parseDatasetFunctionSimple(exampleProto):
            return tf.io.parse_single_example(exampleProto, datasetFeatures)
        
        dataset = dataset.map(parseDatasetFunctionSimple, num_parallel_calls=4)

        for d in dataset:
            data_buffer.append(d)

    logging.info(f'Loaded {len(data_buffer)} sentences.')

    if normalize and len(data_buffer) > 0:
        features = [d['inputFeatures'] for d in data_buffer]
        features = np.concatenate(features, axis=0)
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        for d in data_buffer:
            d['inputFeatures'] = (d['inputFeatures'] - mean) / (std + 1e-8)

    #dataset_gen = data_generator(data_buffer)
    # dataset = tf.data.Dataset.from_generator(lambda dataset_gen,
    #                                         output_types={
    #                                             'inputFeatures': tf.float32,
    #                                             'seqClassIDs': tf.int64,
    #                                             'nTimeSteps': tf.int64,
    #                                             'nSeqElements': tf.int64,
    #                                             'transcription': tf.int64
    #                                         })
    return data_buffer


def clean_label(label, task):
    if task == 'handwriting':
        return label.replace(' ', '')
    
    elif task == 'speech':
        # Remove punctuation
        label = re.sub(r'[^a-zA-Z\- \']', '', label)
        label = label.replace('--', '').lower()
        label = label.replace(" '", "'").lower()

        label = label.strip()
        label = ' '.join(label.split())

        return label
    

def get_phonemes(label, prepend_sil=False):
        g2p = G2p()

        # Change 'a' to 'ay' if we are in spelling mode to correct phonemization
        if all(len(word) == 1 for word in label.split()):
            label = label.replace('a','ay')

        # Convert to phonemes
        phonemes = []
        if len(label) == 0:
            phonemes = SIL_DEF
        else:
            if prepend_sil:
                phonemes.append('SIL')
            for p in g2p(label):
                if p==' ':
                    phonemes.append('SIL')

                p = re.sub(r'[0-9]', '', p)  # Remove stress
                if re.match(r'[A-Z]+', p):  # Only keep phonemes
                    phonemes.append(p)

            #add one SIL symbol at the end so there's one at the end of each word
            phonemes.append('SIL')
            
        return phonemes